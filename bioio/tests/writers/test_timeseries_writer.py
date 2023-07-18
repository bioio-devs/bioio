#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from typing import Callable, Tuple

import bioio_base as biob
import imageio
import numpy as np
import pytest

from bioio.writers.timeseries_writer import TimeseriesWriter
from bioio.writers.two_d_writer import TwoDWriter

from ..conftest import array_constructor


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, read_shape",
    [
        # TODO: Failing currently, needs work,
        # see https://github.com/bioio-devs/bioio/issues/10
        # ((30, 100, 100), None, (30, 100, 100)),
        # Note that files get saved out with RGBA, instead of just RGB
        ((30, 100, 100, 3), None, (30, 100, 100, 4)),
        # TODO: Failing currently, needs work,
        # see https://github.com/bioio-devs/bioio/issues/10
        # ((100, 30, 100), "XTY", (30, 100, 100)),
        # Note that files get saved out with RGBA, instead of just RGB
        ((3, 100, 30, 100), "SYTX", (30, 100, 100, 4)),
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
@pytest.mark.parametrize("filename", ["e.gif"])
def test_timeseries_writer(
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

    # TODO: Actually uncovered bug in DefaultReader :(
    # dask_a = aicsimageio.readers.default_reader.DefaultReader(save_uri).dask_data
    # data = dask_a.compute()
    # assert data.shape == read_shape
    # assert reader.shape[-1] <

    # Read written result and check basics
    fs, path = biob.io.pathlike_to_fs(save_uri)
    extension, mode = TwoDWriter.get_extension_and_mode(path)
    with fs.open(path) as open_resource:
        with imageio.get_reader(open_resource, format=extension, mode=mode) as reader:
            # Read and stack all frames
            frames = []
            for frame in reader:
                print(frame.shape)
                frames.append(frame)

            data = np.stack(frames)

            # Check basics
            assert data.shape == read_shape
            assert data.shape[-1] <= 4

            # Can't do "easy" testing because compression + shape mismatches on RGB data


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
