from importlib import metadata
from importlib.metadata import EntryPoint
from typing import Callable, Iterable

import pytest

from .conftest import TestWriterSpec


def test_dummy_writer_discovery_and_api(
    writer_factory: Callable[[Iterable[TestWriterSpec]], list[EntryPoint]],
) -> None:
    # Arrange: synthetic DummyWriter
    specs = [
        TestWriterSpec(
            name="DummyWriter",
        )
    ]
    writer_factory(specs)

    # Entry-point registration
    eps = metadata.entry_points(group="bioio.writers")
    names = {ep.name for ep in eps}
    assert "DummyWriter" in names, f"found: {sorted(names)}"

    # Public API: Import to check register
    import bioio.writers as pkg

    assert "DummyWriter" in getattr(pkg, "__all__", [])

    # Import works
    from bioio.writers import DummyWriter  # type: ignore[attr-defined]

    assert DummyWriter.__name__ == "DummyWriter"


def test_dummy_writer_save_stub(
    writer_factory: Callable[[Iterable[TestWriterSpec]], list[EntryPoint]],
) -> None:
    # Arrange: synthetic DummyWriter
    specs = [
        TestWriterSpec(
            name="DummyWriter",
            raises_on_save=True,
            save_message="Dummy writer stub",
        )
    ]
    writer_factory(specs)

    from bioio.writers import DummyWriter  # type: ignore[attr-defined]

    # Should raise our stub error
    with pytest.raises(NotImplementedError):
        DummyWriter.save(data=[1], uri="unused", dim_order="XYZ")
