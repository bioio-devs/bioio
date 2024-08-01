#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from io import StringIO

import numpy as np

import bioio

from ..plugins import dump_plugins
from .conftest import DUMMY_PLUGIN_NAME, DUMMY_PLUGIN_PATH, InstallPackage


def test_dump_plugins() -> None:
    with InstallPackage(package_path=DUMMY_PLUGIN_PATH, package_name=DUMMY_PLUGIN_NAME):
        # Capture the output of dump_plugins
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            dump_plugins()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Check if package name is in the output
        assert DUMMY_PLUGIN_NAME in output


def test_plugin_feasibility_report() -> None:
    # Arrange
    test_image = np.random.rand(10, 10)
    expected_error_msg = "missing 1 required positional argument: 'path'"
    with InstallPackage(package_path=DUMMY_PLUGIN_PATH, package_name=DUMMY_PLUGIN_NAME):
        # Act
        actual_output = bioio.plugin_feasibility_report(test_image)
        # Assert
        assert actual_output["ArrayLike"].supported is True
        assert actual_output["ArrayLike"].error is None
        assert actual_output["dummy-plugin"].supported is False
        assert expected_error_msg in (actual_output["dummy-plugin"].error or "")
