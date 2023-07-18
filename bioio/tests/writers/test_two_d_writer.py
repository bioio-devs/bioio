#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from typing import Callable, Tuple

import bioio_base as biob
import imageio
import numpy as np
import pytest

from bioio.writers.two_d_writer import TwoDWriter

from ..conftest import array_constructor


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, read_shape",
    [
        ((100, 100, 3), None, (100, 100, 3)),
        ((100, 100), None, (100, 100)),
        ((100, 100), "XY", (100, 100)),
        ((3, 100, 100), "SYX", (100, 100, 3)),
        ((100, 3, 100), "XSY", (100, 100, 3)),
        pytest.param(
            (1, 1, 1, 1),
            None,
            None,
            marks=pytest.mark.xfail(raises=biob.exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1),
            None,
            None,
            marks=pytest.mark.xfail(raises=biob.exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            "STCZYX",
            None,
            marks=pytest.mark.xfail(raises=biob.exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1),
            "AB",
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["a.png", "d.bmp"])
def test_two_d_writer(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    write_dim_order: str,
    read_shape: Tuple[int, ...],
    filename: str,
    tmp_path: pathlib.Path,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = tmp_path / filename

    # Save
    TwoDWriter.save(arr, save_uri, write_dim_order)

    # Read written result and check basics
    fs, path = biob.io.pathlike_to_fs(save_uri)
    extension, mode = TwoDWriter.get_extension_and_mode(path)
    with fs.open(path) as open_resource:
        with imageio.get_reader(open_resource, format=extension, mode=mode) as reader:
            data = np.asarray(reader.get_data(0))

            # Check basics
            assert data.shape == read_shape
