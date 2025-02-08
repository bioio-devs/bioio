#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration for tests! There are a whole list of hooks you can define in this file to
run before, after, or to mutate how tests run. Commonly for most of our work, we use
this file to define top level fixtures that may be needed for tests throughout multiple
test files.

In this case, while we aren't using this fixture in our tests, the prime use case for
something like this would be when we want to preload a file to be used in multiple
tests. File reading can take time, so instead of re-reading the file for each test,
read the file once then use the loaded content.

Docs: https://docs.pytest.org/en/latest/example/simple.html
      https://docs.pytest.org/en/latest/plugins.html#requiring-loading-plugins-in-a-test-module-or-conftest-file
"""
import importlib
import logging
import pathlib
import subprocess
import sys
import typing

import dask.array as da
import numpy as np
import pytest

import bioio

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@pytest.fixture
def sample_text_file(
    tmp_path: pathlib.Path,
) -> typing.Generator[pathlib.Path, None, None]:
    example_file = tmp_path / "temp-example.txt"
    example_file.write_text("just some example text here")
    yield example_file


def np_random_from_shape(
    shape: typing.Tuple[int, ...], **kwargs: typing.Any
) -> np.ndarray:
    return np.random.randint(255, size=shape, **kwargs)


def da_random_from_shape(
    shape: typing.Tuple[int, ...], **kwargs: typing.Any
) -> da.Array:
    return da.random.randint(255, size=shape, **kwargs)


array_constructor = pytest.mark.parametrize(
    "array_constructor",
    [np_random_from_shape, da_random_from_shape],
)

DUMMY_PLUGIN_NAME = "dummy-plugin"
DUMMY_PLUGIN_PATH = pathlib.Path(__file__).parent / DUMMY_PLUGIN_NAME


class InstallPackage:
    def __init__(
        self, package_path: typing.Union[str, pathlib.Path], package_name: str
    ):
        if isinstance(package_path, pathlib.Path):
            self.package_path = str(package_path)
        else:
            self.package_path = package_path
        self.package_name = package_name

    def __enter__(self) -> "InstallPackage":
        # Install the plugin
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", self.package_path]
        )
        importlib.reload(bioio)
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[type],
        exc_value: typing.Optional[BaseException],
        traceback: typing.Optional[typing.Type[typing.Any]],
    ) -> None:
        # Uninstall the plugin
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", self.package_name]
        )


@pytest.fixture
def dummy_plugin() -> typing.Generator[str, None, None]:
    with InstallPackage(package_path=DUMMY_PLUGIN_PATH, package_name=DUMMY_PLUGIN_NAME):
        yield DUMMY_PLUGIN_NAME
