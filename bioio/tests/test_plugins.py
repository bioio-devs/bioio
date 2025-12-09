#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from io import StringIO
from typing import Any, Dict, List

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

import bioio
import bioio.bio_image as bio_image
from bioio import BioImage

from ..plugins import dump_plugins, get_plugins


def test_dump_plugins(dummy_plugin: str) -> None:
    # Capture the output of dump_plugins
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # Disable plugin caching to prevent other tests from interfering
        dump_plugins(use_cache=False)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # Check if package name is in the output
    assert dummy_plugin in output


def test_plugin_feasibility_report(dummy_plugin: str) -> None:
    # Arrange
    test_image = np.random.rand(10, 10)
    expected_error_msg = "missing 1 required positional argument: 'path'"

    # Act
    actual_output = bioio.plugin_feasibility_report(test_image)

    # Assert
    assert actual_output["ArrayLike"].supported is True
    assert actual_output["ArrayLike"].error is None
    assert actual_output[dummy_plugin].supported is False
    assert expected_error_msg in (actual_output[dummy_plugin].error or "")


@pytest.mark.parametrize(
    "priority_order, expected_first",
    [
        # Case 1: AcceptingReader has higher priority than DummyReader
        (["accepting", "dummy"], "accepting"),
        # Case 2: DummyReader has higher priority than AcceptingReader
        (["dummy", "accepting"], "dummy"),
    ],
)
def test_bioimage_plugin_priority_modulates_reader(
    monkeypatch: MonkeyPatch,
    dummy_plugin: str,
    accepting_plugin: str,
    priority_order: List[str],
    expected_first: str,
) -> None:
    from accepting_plugin import Reader as AcceptingReader
    from dummy_plugin import Reader as DummyReader

    # Arrange
    plugins_by_ext = get_plugins(use_cache=False)

    dummy_entry = None
    accepting_entry = None
    for plugin_list in plugins_by_ext.values():
        for plugin in plugin_list:
            reader_cls = plugin.metadata.get_reader()
            if reader_cls is DummyReader:
                dummy_entry = plugin
            elif reader_cls is AcceptingReader:
                accepting_entry = plugin

    # Build a controlled plugin mapping where both readers claim the same extension
    # and where the base order is [Dummy, Accepting]. This lets us see exactly how
    # the reader list changes which reader BioImage selects.
    test_ext = ".czi"
    fake_plugins_by_ext = {test_ext: [dummy_entry, accepting_entry]}

    # Patch the module-level get_plugins used inside BioImage.determine_plugin so
    # it sees only our controlled mapping.
    monkeypatch.setattr(
        bio_image,
        "get_plugins",
        lambda use_cache: fake_plugins_by_ext,
    )

    # Patch is_supported_image on BOTH readers so they do not hit the base-class
    # path handling (which would enforce file existence and raise FileNotFoundError).
    # We want both readers to *claim* they support this test path.
    def always_supported(
        cls: type[Any],
        image: Any,
        fs_kwargs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        return True

    monkeypatch.setattr(
        DummyReader,
        "is_supported_image",
        classmethod(always_supported),
    )
    monkeypatch.setattr(
        AcceptingReader,
        "is_supported_image",
        classmethod(always_supported),
    )

    # Patch DummyReader.__init__ so using it doesn't raise NotImplementedError.
    def dummy_init(
        self: Any,
        image: Any,
        fs_kwargs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._fs = None
        self._path = str(image)

    monkeypatch.setattr(DummyReader, "__init__", dummy_init)

    # Map the parameterized string names to the actual Reader classes.
    name_to_cls: Dict[str, type[Any]] = {
        "dummy": DummyReader,
        "accepting": AcceptingReader,
    }

    reader_priority = [name_to_cls[name] for name in priority_order]
    expected_cls = name_to_cls[expected_first]
    test_path = f"test{test_ext}"

    # Act: pass the priority list via `reader=` (new API)
    img = BioImage(test_path, reader=reader_priority)

    # Assert
    assert isinstance(img.reader, expected_cls)
