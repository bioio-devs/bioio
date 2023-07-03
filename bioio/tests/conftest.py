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

import pathlib
import typing

import pytest

LOCAL_RESOURCES_DIR = pathlib.Path(__file__).parent / "resources"
LOCAL_RESOURCES_WRITE_DIR = pathlib.Path(__file__).parent / "writer_products"


def get_resource_full_path(filename: str) -> typing.Union[str, pathlib.Path]:
    return LOCAL_RESOURCES_DIR / filename


@pytest.fixture
def data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
