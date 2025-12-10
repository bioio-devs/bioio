from importlib.metadata import EntryPoint
from typing import Callable, Iterable

import numpy as np
from pytest import CaptureFixture

import bioio

from ..plugins import dump_plugins, get_plugins
from .conftest import TestPluginSpec


def test_dump_plugins(
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
    capsys: CaptureFixture[str],
) -> None:
    # Arrange: create synthetic plugin
    specs = [
        TestPluginSpec(
            name="dummy_plugin",
            supported_extensions=[".txt"],
        )
    ]
    plugin_factory(specs)

    # Act
    dump_plugins(use_cache=False)

    # Capture stdout
    captured = capsys.readouterr()
    output = captured.out

    # Assert: plugin name appears in dump
    assert "dummy_plugin" in output


def test_plugin_feasibility_report(
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
) -> None:
    # Arrange: synthetic dummy plugin that FAILS in is_supported_image
    expected_error_msg = "This reader does not support"

    specs = [
        TestPluginSpec(
            name="dummy_plugin",
            supported_extensions=[".txt"],
            fail_on_is_supported=True,
            fail_message=expected_error_msg,
        )
    ]
    plugin_factory(specs)

    test_image = np.random.rand(10, 10)

    # Act
    report = bioio.plugin_feasibility_report(test_image)

    # Assert (1): built-in ArrayLikeReader reads it
    assert report["ArrayLike"].supported is True
    assert report["ArrayLike"].error is None

    # Assert (2): our synthetic dummy plugin fails
    dummy_state = report["dummy_plugin"]
    assert dummy_state.supported is False
    assert dummy_state.error is not None
    assert expected_error_msg in dummy_state.error


def test_get_plugins_orders_extension_keys_by_descending_length(
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
) -> None:
    # Arrange: fake plugins that support tiff files.
    specs = [
        TestPluginSpec(
            name="plugin_ome_tiff",
            supported_extensions=[".ome.tiff"],
        ),
        TestPluginSpec(
            name="plugin_tiff",
            supported_extensions=[".tiff"],
        ),
        TestPluginSpec(
            name="plugin_ome_tif",
            supported_extensions=[".ome.tif"],
        ),
        TestPluginSpec(
            name="plugin_tif",
            supported_extensions=[".tif"],
        ),
    ]
    plugin_factory(specs)

    # Act: get_plugins orders
    plugins_by_ext = get_plugins(use_cache=False)
    keys = list(plugins_by_ext.keys())

    # Assert: Keys are in right order
    assert keys == [".ome.tiff", ".ome.tif", ".tiff", ".tif"]


def test_get_plugins_orders_plugins_by_family_count_and_raw_ext_count(
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
) -> None:
    # Arrange: fake plugins that support tiff files.
    specs = [
        TestPluginSpec(
            name="focused_tif",
            supported_extensions=[".tif"],
        ),
        # Same family as .tif (".ome.tif" endswith ".tif"), but more raw exts
        TestPluginSpec(
            name="ome_tif_family",
            supported_extensions=[".ome.tif", ".tif"],
        ),
        # Two unrelated families: ".tif" and ".tiff"
        TestPluginSpec(
            name="tif_and_tiff_two_families",
            supported_extensions=[".tif", ".tiff"],
        ),
    ]
    plugin_factory(specs)

    # Act: get_plugins orders
    plugins_by_ext = get_plugins(use_cache=False)
    tif_plugins = plugins_by_ext[".tif"]
    tif_plugin_names = [p.entrypoint.name for p in tif_plugins]

    # Expected:
    #   1. focused_tif               (family_count=1, raw_ext_count=1)
    #   2. ome_tif_family            (family_count=1, raw_ext_count=2)
    #   3. tif_and_tiff_two_families (family_count=2, raw_ext_count=2)
    assert tif_plugin_names == [
        "focused_tif",
        "ome_tif_family",
        "tif_and_tiff_two_families",
    ]


def test_get_plugins_orders_plugins_alphabetically_on_tie(
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
) -> None:
    # Arrange: plugins with same support but different names.
    specs = [
        TestPluginSpec(
            name="z_plugin",
            supported_extensions=[".tif"],
        ),
        TestPluginSpec(
            name="a_plugin",
            supported_extensions=[".tif"],
        ),
    ]
    plugin_factory(specs)

    # Act: get_plugins orders
    plugins_by_ext = get_plugins(use_cache=False)
    tif_plugins = plugins_by_ext[".tif"]
    tif_plugin_names = [p.entrypoint.name for p in tif_plugins]

    # Assert: alphabetical
    assert tif_plugin_names == ["a_plugin", "z_plugin"]
