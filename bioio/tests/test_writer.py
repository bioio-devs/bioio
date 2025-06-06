from importlib.metadata import entry_points

import pytest


def test_dummy_writer_discovery_and_api(dummy_plugin: str) -> None:
    # Entry-point registration
    eps = entry_points(group="bioio.writers")
    names = {ep.name for ep in eps}
    assert "DummyWriter" in names, f"found: {sorted(names)}"

    # Public API (__all__)
    import bioio.writers as pkg

    assert "DummyWriter" in getattr(pkg, "__all__", [])

    # Import works (dynamic, so ignore mypy attr-defined error)
    from bioio.writers import DummyWriter  # type: ignore[attr-defined]

    assert DummyWriter.__name__ == "DummyWriter"


def test_dummy_writer_save_stub(dummy_plugin: str) -> None:
    # The save() stub should raise NotImplementedError
    from bioio.writers import DummyWriter  # type: ignore[attr-defined]

    with pytest.raises(NotImplementedError):
        DummyWriter.save(data=[1], uri="unused", dim_order="XYZ")
