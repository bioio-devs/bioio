import sys
import types
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import Any, Callable, Iterable, List, Optional, Type

import bioio_base
import pytest
from _pytest.monkeypatch import MonkeyPatch
from bioio_base.reader import Reader as BaseReader

import bioio.plugins as plugins

# -----------------------------------------------------------------------------
# Plugin specification
# -----------------------------------------------------------------------------


@dataclass
class TestPluginSpec:
    """
    Declarative specification for a synthetic BioIO reader plugin.

    Field semantics
    ---------------
    name:
        Logical plugin name. Used for the EntryPoint name, the Reader.NAME
        identifier, and (by default) the synthetic module name.

    supported_extensions:
        File extensions the plugin *claims* to support. These drive extension-
        based plugin discovery and ordering in BioIO.

    fail_on_is_supported:
        If True, the synthetic Reader’s `_is_supported_image` method will raise
        a TypeError instead of returning a boolean.

    fail_message:
        Error message used when raising from `_is_supported_image` or `__init__`.

    check_anon_in_init:
        If True, the synthetic Reader’s `__init__` will require
        `fs_kwargs["anon"] == True` and raise otherwise.
    """

    name: str
    supported_extensions: List[str]
    fail_on_is_supported: bool = False
    fail_message: str = "missing 1 required positional argument: 'path'"
    check_anon_in_init: bool = False


# -----------------------------------------------------------------------------
# Minimal ReaderMetadata shim
# -----------------------------------------------------------------------------


class _TestReaderMetadata:
    """
    Minimal stand-in for real ReaderMetadata.

    Only implements the API surface used by bioio.plugins.
    """

    def __init__(
        self,
        name: str,
        exts: List[str],
        reader_cls: Type[BaseReader],
    ) -> None:
        self._name = name
        self._exts = list(exts)
        self._reader_cls = reader_cls

    def get_supported_extensions(self) -> List[str]:
        return list(self._exts)

    def get_reader(self) -> Type[BaseReader]:
        return self._reader_cls


# -----------------------------------------------------------------------------
# Synthetic Reader construction
# -----------------------------------------------------------------------------


def _make_reader_class(spec: TestPluginSpec) -> Type[BaseReader]:
    """
    Create a Reader subclass whose behavior is controlled by the spec.

    The reader does not perform IO; it only participates in plugin selection
    and support checks.
    """

    class TestReader(BaseReader):
        NAME = spec.name

        def __init__(
            self,
            image: Any,
            fs_kwargs: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            fs_kwargs = fs_kwargs or {}

            # Optional invariant for tests that expect anon=True (S3)
            if spec.check_anon_in_init and not fs_kwargs.get("anon", False):
                raise bioio_base.exceptions.UnsupportedFileFormatError(
                    "test", "test", spec.fail_message
                )

        @property
        def scenes(self) -> list[str]:
            return ["Scene 0"]

        def _read_immediate(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        def _read_delayed(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        @classmethod
        def _is_supported_image(
            cls,
            image: Any,
            fs_kwargs: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> bool:
            # Used to simulate failing or buggy support checks
            if spec.fail_on_is_supported:
                raise TypeError(spec.fail_message)
            return True

    TestReader.__name__ = f"{spec.name.title().replace('-', '_')}Reader"
    return TestReader


# -----------------------------------------------------------------------------
# Plugin "installation" helpers
# -----------------------------------------------------------------------------


def _install_ephemeral_module(spec: TestPluginSpec) -> str:
    """
    Register a synthetic plugin module in sys.modules.

    This simulates an installed plugin package without touching disk.
    """
    module_name = f"bioio_test_plugins.{spec.name}"
    mod = types.ModuleType(module_name)

    reader_cls = _make_reader_class(spec)
    reader_metadata = _TestReaderMetadata(
        name=spec.name,
        exts=spec.supported_extensions,
        reader_cls=reader_cls,
    )

    setattr(mod, "Reader", reader_cls)
    setattr(mod, "ReaderMetadata", reader_metadata)
    sys.modules[module_name] = mod

    return module_name


def _make_entry_points(specs: Iterable[TestPluginSpec]) -> List[EntryPoint]:
    """
    Create EntryPoint objects pointing at the synthetic plugin modules.
    """
    eps: List[EntryPoint] = []
    for spec in specs:
        module_name = _install_ephemeral_module(spec)
        eps.append(
            EntryPoint(
                name=spec.name,
                group="bioio.readers",
                value=module_name,
            )
        )
    return eps


# -----------------------------------------------------------------------------
# Public factory fixture
# -----------------------------------------------------------------------------


@pytest.fixture
def plugin_factory(
    monkeypatch: MonkeyPatch,
) -> Callable[[Iterable[TestPluginSpec]], List[EntryPoint]]:
    """
    Factory fixture that installs synthetic reader plugins for a test.

    What it does
    ------------
    Calling the returned `factory(...)` function will:

    1) Create *in-memory* plugin modules (no files on disk)
       - For each TestPluginSpec, we create a `types.ModuleType` and register it
         in `sys.modules` under a module name like `bioio_test_plugins.<name>`.
       - Each module exports:
           - `Reader`: a synthetic `BaseReader` subclass
           - `ReaderMetadata`: a minimal metadata object that reports
             `supported_extensions` and returns the `Reader` class.

    2) Fabricate entry points for those modules
       - We build `EntryPoint(group="bioio.readers", value="<module_name>")`
         for each synthetic plugin.

    3) Patch BioIO discovery to return only these plugins
       - `bioio.plugins.entry_points(group="bioio.readers")` is monkeypatched
         to return the fabricated entry points, so BioIO “discovers” them as if
         they were installed distributions.

    Call `plugin_factory(...)` before any code path that triggers plugin
    discovery (e.g. `BioImage(...)`, `get_plugins(...)`, etc.), otherwise the
    plugin cache might already be populated.

    Creating Reader Specifications
    -------------------------------
    Reader behavior is configured via `TestPluginSpec`, which describes how each
    synthetic plugin should behave during discovery and support checks.

    Minimal specification:
        TestPluginSpec(
            name="dummy",
            supported_extensions=[".txt"],
        )

    Advanced options:
        - fail_on_is_supported:
            Causes the reader’s `_is_supported_image` method to raise instead of
            returning a boolean. This is used to simulate buggy or incompatible
            plugins and to test fallback behavior when support checks fail.

        - fail_message:
            Error message used when raising from `_is_supported_image` or
            `__init__`. Allows tests to assert on error content.

        - check_anon_in_init:
            Requires `fs_kwargs["anon"] == True` in the reader constructor.
            This is primarily used to test correct handling of public S3 URLs
            and propagation of `anon=True` through BioIO.

    Examples
    --------
    Multiple plugins registration):
        def test_two_plugins_same_ext(plugin_factory, sample_text_file):
            plugin_factory([
                TestPluginSpec(name="a", supported_extensions=[".txt"]),
                TestPluginSpec(name="b", supported_extensions=[".tiff"]),
            ])
            img = BioImage(sample_text_file)
    """

    def factory(specs: Iterable[TestPluginSpec]) -> List[EntryPoint]:
        created_eps = _make_entry_points(specs)

        # Force BioIO to see only these synthetic reader plugins
        def fake_entry_points(
            group: Optional[str] = None,
            **kwargs: Any,
        ) -> List[EntryPoint]:
            return created_eps if group == "bioio.readers" else []

        monkeypatch.setattr(plugins, "entry_points", fake_entry_points)

        # Disable dependency-version filtering in tests
        monkeypatch.setattr(
            plugins,
            "get_dependency_version_range_for_distribution",
            lambda *_: (None, None),
        )

        # Ensure a clean plugin resolution per test
        plugins.plugins_by_ext_cache.clear()
        return created_eps

    return factory
