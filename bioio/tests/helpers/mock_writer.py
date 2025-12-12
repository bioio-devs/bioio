import sys
import types
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import Any, Callable, Iterable, List, Type

import bioio_base
import pytest
from _pytest.monkeypatch import MonkeyPatch

# -----------------------------------------------------------------------------
# Writer specification
# -----------------------------------------------------------------------------


@dataclass
class TestWriterSpec:
    """
    Declarative specification for a synthetic BioIO writer.

    Field semantics
    ---------------
    name:
        Writer class name and EntryPoint name. This also becomes the symbol that
        should appear in `bioio.writers.__all__` after reloading the package.

    raises_on_save:
        If True, the synthetic Writer.save(...) raises NotImplementedError.
        This is useful for tests that only need discovery + API wiring without
        performing real writes.

    save_message:
        Message used for the NotImplementedError. Allows tests to assert on
        error content when desired.
    """

    name: str
    raises_on_save: bool = True
    save_message: str = "Dummy writer stub"


# -----------------------------------------------------------------------------
# Synthetic Writer construction
# -----------------------------------------------------------------------------


def _make_writer_class(spec: TestWriterSpec) -> Type[bioio_base.writer.Writer]:
    """
    Create a Writer subclass whose save(...) behavior is controlled by the spec.
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


# -----------------------------------------------------------------------------
# Writer "installation" helpers
# -----------------------------------------------------------------------------


def _install_writer_module(spec: TestWriterSpec) -> tuple[str, str]:
    """
    Register a synthetic writer module in sys.modules.

    This simulates an installed writer module without touching disk. The module
    exports a single symbol named `spec.name` (e.g. `DummyWriter`).
    """
    module_name = f"bioio_test_writers.{spec.name.lower()}"
    mod = types.ModuleType(module_name)

    writer_cls = _make_writer_class(spec)
    setattr(mod, spec.name, writer_cls)

    sys.modules[module_name] = mod
    return module_name, spec.name


def _make_writer_entry_points(specs: Iterable[TestWriterSpec]) -> List[EntryPoint]:
    """
    Create EntryPoint objects pointing at the synthetic writer modules.

    Writer entry points use the "module:object" form (e.g. "x.y:DummyWriter").
    """
    eps: List[EntryPoint] = []
    for spec in specs:
        module_name, class_name = _install_writer_module(spec)
        eps.append(
            EntryPoint(
                name=spec.name,
                group="bioio.writers",
                value=f"{module_name}:{class_name}",
            )
        )
    return eps


# -----------------------------------------------------------------------------
# Public factory fixture
# -----------------------------------------------------------------------------


@pytest.fixture
def writer_factory(
    monkeypatch: MonkeyPatch,
) -> Callable[[Iterable[TestWriterSpec]], List[EntryPoint]]:
    """
    Factory fixture that installs synthetic writers for a test.

    What it does
    ------------
    Calling the returned `factory(...)` function will:

    1) Create *in-memory* writer modules (no files on disk)
       - For each TestWriterSpec, we create a `types.ModuleType` and register it
         in `sys.modules` under `bioio_test_writers.<lower(name)>`.
       - Each module exports a writer class symbol with the exact name provided
         by `spec.name` (e.g. `DummyWriter`).

    2) Fabricate writer entry points
       - We build `EntryPoint(group="bioio.writers", value="module:Class")`
         for each synthetic writer.

    3) Patch Python entry point discovery for the writer group
       - We monkeypatch `importlib.metadata.entry_points` (or the backport) so
         `metadata.entry_points(group="bioio.writers")` returns only our synthetic
         writers.

    4) Reload `bioio.writers`
       - The writers package typically populates its public API at import time.
         Reloading ensures it re-reads the patched entry points and updates
         `__all__` accordingly.

    Creating Writer Specifications
    ------------------------------
    Minimal specification (discovery/API only):
        TestWriterSpec(name="DummyWriter")

    Save-stub behavior:
        TestWriterSpec(
            name="DummyWriter",
            raises_on_save=True,
            save_message="Dummy writer stub",
        )

    Examples
    --------
        def test_writer_discovery(writer_factory):
            writer_factory([TestWriterSpec(name="DummyWriter")])

            from importlib import metadata
            assert "DummyWriter" in {
                ep.name for ep in metadata.entry_points(group="bioio.writers")
            }

            from bioio.writers import DummyWriter
            assert DummyWriter.__name__ == "DummyWriter"
    """

    def factory(specs: Iterable[TestWriterSpec]) -> List[EntryPoint]:
        created_eps = _make_writer_entry_points(specs)

        def fake_entry_points(
            group: str | None = None, **kwargs: Any
        ) -> List[EntryPoint]:
            return created_eps if group == "bioio.writers" else []

        # Patch entry_points on stdlib importlib.metadata (or its backport)
        try:
            import importlib.metadata as importlib_metadata

            monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points)
        except ImportError:  # pragma: no cover
            import importlib_metadata as metadata_backport  # type: ignore

            monkeypatch.setattr(metadata_backport, "entry_points", fake_entry_points)

        # Reload so bioio.writers re-registers from our patched entry points
        import importlib as _importlib

        import bioio.writers as writers_mod

        _importlib.reload(writers_mod)
        return created_eps

    return factory
