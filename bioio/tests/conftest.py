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

import logging
import pathlib
import typing

import dask.array as da
import numpy as np
import pytest

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@pytest.fixture
def sample_text_file(tmp_path: pathlib.Path) -> pathlib.Path:
    example_file = tmp_path / "temp-example.txt"
    example_file.write_text("just some example text here")
    return example_file


def np_random_from_shape(
    shape: typing.Tuple[int, ...], **kwargs: typing.Any
) -> np.ndarray:
    return np.random.randint(255, size=shape, **kwargs)


def da_random_from_shape(
    shape: typing.Tuple[int, ...], **kwargs: typing.Any
) -> da.Array:
    return da.random.randint(255, size=shape, **kwargs)


array_constructor = pytest.mark.parametrize(
    "array_constructor", [np_random_from_shape, da_random_from_shape]
)
