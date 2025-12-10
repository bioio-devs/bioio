import logging
import pathlib
import sys
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple, Type

import bioio_base
import dask.array as da
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from bioio_base.reader import Reader as BaseReader

import bioio.plugins as plugins

# EntryPoint import with fallback for Python < 3.10
try:
    from importlib.metadata import EntryPoint
except ImportError:
    from importlib_metadata import EntryPoint  # type: ignore

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@pytest.fixture
def sample_text_file(
    tmp_path: pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    example_file = tmp_path / "temp-example.txt"
    example_file.write_text("just some example text here")
    yield example_file


def np_random_from_shape(shape: typing.Tuple[int, ...], **kwargs: Any) -> np.ndarray:
    return np.random.randint(255, size=shape, **kwargs)


def da_random_from_shape(shape: typing.Tuple[int, ...], **kwargs: Any) -> da.Array:
    return da.random.randint(255, size=shape, **kwargs)


array_constructor = pytest.mark.parametrize(
    "array_constructor",
    [np_random_from_shape, da_random_from_shape],
)


@dataclass
class TestPluginSpec:
    """
    Configuration for a single synthetic test plugin.
    """

    name: str
    supported_extensions: List[str]
    module_name: Optional[str] = None
    fail_on_is_supported: bool = False
    fail_message: str = "missing 1 required positional argument: 'path'"
    check_anon_in_init: bool = False


class _TestReaderMetadata:
    """
    Synthetic metadata object that matches what plugins.py uses.
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


def _make_reader_class(spec: TestPluginSpec) -> Type[BaseReader]:
    """
    Create Reader subclass for synthetic plugin.
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

            # If requested, enforce anon=True (for S3 tests)
            if spec.check_anon_in_init and not fs_kwargs.get("anon", False):
                raise bioio_base.exceptions.UnsupportedFileFormatError(
                    "test", "test", spec.fail_message
                )

        @property
        def scenes(self) -> list[str]:
            return ["Scene 0"]

        def _read_immediate(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("_read_immediate not used in tests")

        def _read_delayed(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("_read_delayed not used in tests")

        @classmethod
        def _is_supported_image(
            cls,
            image: Any,
            fs_kwargs: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> bool:
            if spec.fail_on_is_supported:
                raise TypeError(spec.fail_message)
            return True

    TestReader.__name__ = f"{spec.name.title().replace('-', '_')}Reader"
    return TestReader


def _install_ephemeral_module(spec: TestPluginSpec) -> str:
    """
    Create a synthetic module in sys.modules exposing ReaderMetadata.
    """
    module_name = spec.module_name or f"bioio_test_plugins.{spec.name}"
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
    eps: List[EntryPoint] = []
    for spec in specs:
        module_name = _install_ephemeral_module(spec)
        ep = EntryPoint(
            name=spec.name,
            group="bioio.readers",
            value=module_name,
        )
        eps.append(ep)
    return eps


@pytest.fixture
def plugin_factory(
    monkeypatch: MonkeyPatch,
) -> Callable[[Iterable[TestPluginSpec]], List[EntryPoint]]:
    """
    Factory fixture to install synthetic plugins.

    Usage:
        specs = [
            TestPluginSpec(
                name="dummy_plugin",
                supported_extensions=[".txt"],
            ),
        ]
        plugin_factory(specs)
    """

    def factory(specs: Iterable[TestPluginSpec]) -> List[EntryPoint]:
        created_eps = _make_entry_points(specs)

        # Patch entry_points()
        def fake_entry_points(
            group: Optional[str] = None,
            **kwargs: Any,
        ) -> List[EntryPoint]:
            if group == "bioio.readers":
                return created_eps
            return []

        monkeypatch.setattr(plugins, "entry_points", fake_entry_points)

        # Patch version-range
        def fake_version_range(
            dist_name: str,
            dep_name: str,
        ) -> tuple[Optional[str], Optional[str]]:
            return (None, None)

        monkeypatch.setattr(
            plugins,
            "get_dependency_version_range_for_distribution",
            fake_version_range,
        )

        # Clear plugin cache so each test starts clean
        plugins.plugins_by_ext_cache.clear()
        return created_eps

    return factory


@dataclass
class TestWriterSpec:
    """
    Configuration for a single synthetic test writer.
    """

    name: str
    module_name: Optional[str] = None
    raises_on_save: bool = True
    save_message: str = "Dummy writer stub"


def _make_writer_class(spec: TestWriterSpec) -> Type[bioio_base.writer.Writer]:
    """
    Synthetic Writer subclass for tests.
    """

    class TestWriter(bioio_base.writer.Writer):
        @staticmethod
        def save(
            data: bioio_base.types.ArrayLike,
            uri: bioio_base.types.PathLike,
            dim_order: str = bioio_base.dimensions.DEFAULT_DIMENSION_ORDER,
            **kwargs: Any,
        ) -> None:
            if spec.raises_on_save:
                raise NotImplementedError(spec.save_message)

    TestWriter.__name__ = spec.name
    return TestWriter


def _install_writer_module(spec: TestWriterSpec) -> Tuple[str, str]:
    """
    Create a synthetic module in sys.modules exposing the writer class.
    """
    module_name = spec.module_name or f"bioio_test_writers.{spec.name.lower()}"
    mod = types.ModuleType(module_name)

    writer_cls = _make_writer_class(spec)
    setattr(mod, spec.name, writer_cls)

    sys.modules[module_name] = mod
    return module_name, spec.name


def _make_writer_entry_points(specs: Iterable[TestWriterSpec]) -> List[EntryPoint]:
    eps: List[EntryPoint] = []
    for spec in specs:
        module_name, class_name = _install_writer_module(spec)
        ep = EntryPoint(
            name=spec.name,
            group="bioio.writers",
            value=f"{module_name}:{class_name}",
        )
        eps.append(ep)
    return eps


@pytest.fixture
def writer_factory(
    monkeypatch: MonkeyPatch,
) -> Callable[[Iterable[TestWriterSpec]], List[EntryPoint]]:
    """
    Factory fixture to install synthetic writers.

    Usage:
        specs = [
            TestWriterSpec(
                name="DummyWriter",
            ),
        ]
        writer_factory(specs)
    """

    def factory(specs: Iterable[TestWriterSpec]) -> List[EntryPoint]:
        created_eps = _make_writer_entry_points(specs)

        def fake_entry_points(
            group: Optional[str] = None,
            **kwargs: Any,
        ) -> List[EntryPoint]:
            if group == "bioio.writers":
                return created_eps
            return []

        # Patch entry_points implementation
        try:
            import importlib.metadata as importlib_metadata

            monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points)
        except ImportError:
            import importlib_metadata as metadata_backport

            monkeypatch.setattr(metadata_backport, "entry_points", fake_entry_points)

        # reimport so it picks up the fake entry points
        import importlib as _importlib

        import bioio.writers as writers_mod

        _importlib.reload(writers_mod)
        return created_eps

    return factory
