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
import shutil
import typing

import dask.array as da
import numpy as np
import pytest

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

LOCAL_RESOURCES_DIR = pathlib.Path(__file__).parent / "resources"
LOCAL_RESOURCES_WRITE_DIR = pathlib.Path(__file__).parent / "writer_products"


def pytest_sessionstart(session: pytest.Session) -> None:
    """
    Called after the Session object has been created and
    before performing collection and entering the run test suite loop.
    """
    if LOCAL_RESOURCES_WRITE_DIR.exists():
        log.warning(
            f"{LOCAL_RESOURCES_WRITE_DIR.absolute} should not exist at "
            "start of tests, deleting now"
        )
        shutil.rmtree(LOCAL_RESOURCES_WRITE_DIR)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """
    Called after whole test suite run finished, right before
    returning the exit status to the system.
    """
    shutil.rmtree(LOCAL_RESOURCES_WRITE_DIR)


@pytest.fixture
def data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"


def get_resource_full_path(filename: str) -> typing.Union[str, pathlib.Path]:
    return LOCAL_RESOURCES_DIR / filename


def get_resource_write_full_path(filename: str) -> typing.Union[str, pathlib.Path]:
    LOCAL_RESOURCES_WRITE_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_RESOURCES_WRITE_DIR / filename


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
