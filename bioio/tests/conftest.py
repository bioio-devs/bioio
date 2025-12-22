import logging
import pathlib
import typing
from typing import Any, Generator

import dask.array as da
import numpy as np
import pytest

log = logging.getLogger(__name__)

# Import helpers for fixtures
from .helpers.mock_reader import TestPluginSpec, plugin_factory  # noqa: E402,F401
from .helpers.mock_writer import TestWriterSpec, writer_factory  # noqa: E402,F401


@pytest.fixture
def sample_text_file(
    tmp_path: pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    example_file = tmp_path / "temp-example.txt"
    example_file.write_text("just some example text here")
    yield example_file


def np_random_from_shape(shape: typing.Tuple[int, ...], **kwargs: Any) -> np.ndarray:
    return np.random.randint(255, size=shape, **kwargs)


def da_random_from_shape(shape: typing.Tuple[int, ...], **kwargs: Any) -> da.Array:
    return da.random.randint(255, size=shape, **kwargs)


array_constructor = pytest.mark.parametrize(
    "array_constructor",
    [np_random_from_shape, da_random_from_shape],
)
