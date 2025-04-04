#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from typing import Callable, Optional, Tuple

import bioio_base as biob
import imageio
import numpy as np
import pytest

from bioio.writers.timeseries_writer import TimeseriesWriter
from bioio.writers.two_d_writer import TwoDWriter

from ..conftest import array_constructor


@pytest.mark.parametrize(
    "write_shape, write_dim_order, expect_error",
    [
        # === Valid cases ===
        # Grayscale GIF (T, Y, X)
        ((30, 100, 100), None, None),
        # RGB GIF (T, Y, X, 3) – imageio might return RGBA
        ((30, 100, 100, 3), None, None),
        # Weird input that needs dim reordering ("SYTX" → TYXS)
        ((3, 100, 30, 100), "SYTX", None),
        # === Invalid shape cases ===
        # Too few dims (should fail)
        pytest.param(
            (1, 1), None, biob.exceptions.UnexpectedShapeError, marks=pytest.mark.xfail
        ),
        # Too many dims (5D)
        pytest.param(
            (1, 1, 1, 1, 1),
            None,
            biob.exceptions.UnexpectedShapeError,
            marks=pytest.mark.xfail,
        ),
        # 6D with valid dim order (still invalid)
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            "STCZYX",
            biob.exceptions.UnexpectedShapeError,
            marks=pytest.mark.xfail,
        ),
        # Invalid dim order
        pytest.param(
            (1, 1, 1, 1),
            "ABCD",
            biob.exceptions.InvalidDimensionOrderingError,
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.gif"])
@array_constructor
def test_timeseries_writer(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    write_dim_order: Optional[str],
    expect_error: Optional[Exception],
    filename: str,
    tmp_path: pathlib.Path,
) -> None:
    arr = array_constructor(write_shape, dtype=np.uint8)
    save_uri = tmp_path / filename

    # Check error case
    if expect_error is not None:
        with pytest.raises(expect_error):
            TimeseriesWriter.save(arr, save_uri, write_dim_order)
        return

    # Save with TimeseriesWriter
    TimeseriesWriter.save(arr, save_uri, write_dim_order)

    # Recreate expected dimension order post-reshape
    n_dims = len(arr.shape)
    if write_dim_order is None:
        n_dims = len(arr.shape)
        dim_order_used = TimeseriesWriter.DIM_ORDERS[n_dims]
    else:
        dim_order_used = write_dim_order.upper()

    expected_order = TimeseriesWriter.DIM_ORDERS[n_dims]
    if dim_order_used != expected_order:
        arr = biob.transforms.reshape_data(
            arr, given_dims=dim_order_used, return_dims=expected_order
        )

    # Read back with imageio
    fs, path = biob.io.pathlike_to_fs(save_uri)
    extension, mode = TwoDWriter.get_extension_and_mode(path)

    with fs.open(path) as open_resource:
        with imageio.get_reader(open_resource, format=extension, mode=mode) as reader:
            frames = [frame for frame in reader]

    data = np.stack(frames)

    # Validate number of frames
    assert (
        data.shape[0] == arr.shape[0]
    ), f"Expected {arr.shape[0]} frames, got {data.shape[0]}"

    # Validate spatial dimensions
    assert (
        data.shape[1] == arr.shape[1]
    ), f"Expected Y={arr.shape[1]}, got {data.shape[1]}"
    assert (
        data.shape[2] == arr.shape[2]
    ), f"Expected X={arr.shape[2]}, got {data.shape[2]}"

    # Validate channel depth (3 or 4 if present)
    if data.ndim == 4:
        assert data.shape[3] in (3, 4), f"Expected 3 or 4 channels, got {data.shape[3]}"


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, read_shape",
    [
        # We use 112 instead of 100 because FFMPEG block size warnings are annoying
        ((30, 112, 112), None, (30, 112, 112, 3)),
        # Note that files get saved out with RGBA, instead of just RGB
        ((30, 112, 112, 3), None, (30, 112, 112, 3)),
        ((112, 30, 112), "XTY", (30, 112, 112, 3)),
        # Note that files get saved out with RGBA, instead of just RGB
        ((3, 112, 30, 112), "SYTX", (30, 112, 112, 3)),
        pytest.param(
            (1, 1),
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
            (1, 1, 1, 1),
            "ABCD",
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["f.mp4"])
def test_timeseries_writer_ffmpeg(
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

    # Normal save
    TimeseriesWriter.save(arr, save_uri, write_dim_order)

    # Read written result and check basics
    fs, path = biob.io.pathlike_to_fs(save_uri)
    extension, mode = TwoDWriter.get_extension_and_mode(path)
    with fs.open(path) as open_resource:
        with imageio.get_reader(open_resource, format=extension, mode=mode) as reader:
            # Read and stack all frames
            frames = []
            for frame in reader:
                frames.append(frame)

            data = np.stack(frames)

            # Check basics
            assert data.shape == read_shape
            assert data.shape[-1] <= 4

            # Can't do "easy" testing because compression + shape mismatches on RGB data
