#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pathlib
from typing import Callable, List, Tuple

import numpy as np
import pytest
from dask import array as da
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from bioio.writers.ome_zarr_writer_2 import (
    DimTuple,
    OmeZarrWriter,
    chunk_size_from_memory_target,
    compute_level_chunk_sizes_zslice,
    compute_level_shapes,
    resize,
)

from ..conftest import array_constructor


@pytest.mark.parametrize(
    "input_shape, dtype, memory_target, expected_chunk_shape",
    [
        ((1, 1, 1, 128, 128), np.uint16, 1024, (1, 1, 1, 16, 16)),
        ((1, 1, 1, 127, 127), np.uint16, 1024, (1, 1, 1, 15, 15)),
        ((1, 1, 1, 129, 129), np.uint16, 1024, (1, 1, 1, 16, 16)),
        ((7, 11, 128, 128, 128), np.uint16, 1024, (1, 1, 8, 8, 8)),
    ],
)
def test_chunk_size_from_memory_target(
    input_shape: DimTuple,
    dtype: np.dtype,
    memory_target: int,
    expected_chunk_shape: DimTuple,
) -> None:
    chunk_shape = chunk_size_from_memory_target(input_shape, dtype, memory_target)
    assert chunk_shape == expected_chunk_shape


def test_resize() -> None:
    d = da.from_array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    output_shape = (1, 1)
    out_d = resize(d, output_shape)
    assert out_d.shape == output_shape


@pytest.mark.parametrize(
    "in_shape, scale_per_level, num_levels, expected_out_shapes",
    [
        (
            (1, 1, 1, 128, 128),
            (1.0, 1.0, 1.0, 2.0, 2.0),
            2,
            [(1, 1, 1, 128, 128), (1, 1, 1, 64, 64)],
        ),
        (
            (1, 1, 256, 1024, 2048),
            (1.0, 1.0, 1.0, 2.0, 2.0),
            3,
            [(1, 1, 256, 1024, 2048), (1, 1, 256, 512, 1024), (1, 1, 256, 256, 512)],
        ),
        (
            (1, 1, 1, 4, 4),
            (1.0, 1.0, 1.0, 2.0, 2.0),
            5,
            [
                (1, 1, 1, 4, 4),
                (1, 1, 1, 2, 2),
                (1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1),
            ],
        ),
    ],
)
def test_compute_level_shapes(
    in_shape: DimTuple,
    scale_per_level: Tuple[float, float, float, float, float],
    num_levels: int,
    expected_out_shapes: List[DimTuple],
) -> None:
    out_shapes = compute_level_shapes(in_shape, scale_per_level, num_levels)
    assert out_shapes == expected_out_shapes


@pytest.mark.parametrize(
    "in_shapes, expected_out_chunk_shapes",
    [
        (
            [
                (512, 4, 100, 1000, 1000),
                (512, 4, 100, 500, 500),
                (512, 4, 100, 250, 250),
            ],
            [(1, 1, 1, 1000, 1000), (1, 1, 4, 500, 500), (1, 1, 16, 250, 250)],
        )
    ],
)
def test_compute_chunk_sizes_zslice(
    in_shapes: List[DimTuple], expected_out_chunk_shapes: List[DimTuple]
) -> None:
    out_chunk_shapes = compute_level_chunk_sizes_zslice(in_shapes)
    assert out_chunk_shapes == expected_out_chunk_shapes


@array_constructor
@pytest.mark.parametrize(
    "shape, num_levels, scaling, expected_shapes",
    [
        (
            (4, 2, 2, 64, 32),  # easy, powers of two
            3,
            (1, 1, 1, 2, 2),  # downscale xy by two
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
        ),
        (
            (4, 2, 2, 8, 6),
            1,  # no downscaling
            (1, 1, 1, 1, 1),
            [(4, 2, 2, 8, 6)],
        ),
        (
            (1, 1, 1, 13, 23),  # start with odd dimensions
            3,
            (1, 1, 1, 2, 2),
            [(1, 1, 1, 13, 23), (1, 1, 1, 6, 11), (1, 1, 1, 3, 5)],
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.zarr"])
def test_write_ome_zarr(
    array_constructor: Callable,
    filename: str,
    shape: DimTuple,
    num_levels: int,
    scaling: Tuple[float, float, float, float, float],
    expected_shapes: List[DimTuple],
    tmp_path: pathlib.Path,
) -> None:
    # TCZYX order, downsampling x and y only
    im = array_constructor(shape, dtype=np.uint8)
    C = shape[1]

    shapes = compute_level_shapes(shape, scaling, num_levels)
    chunk_sizes = compute_level_chunk_sizes_zslice(shapes)

    # Create an OmeZarrWriter object
    writer = OmeZarrWriter()

    # Initialize the store. Use s3 url or local directory path!
    save_uri = tmp_path / filename
    writer.init_store(str(save_uri), shapes, chunk_sizes, im.dtype)

    # Write the image
    writer.write_t_batches_array(im, channels=[], tbatch=4)

    # TODO: get this from source image
    physical_scale = {
        "c": 1.0,  # default value for channel
        "t": 1.0,
        "z": 1.0,
        "y": 1.0,
        "x": 1.0,
    }
    physical_units = {
        "x": "micrometer",
        "y": "micrometer",
        "z": "micrometer",
        "t": "minute",
    }
    meta = writer.generate_metadata(
        image_name="TEST",
        channel_names=[f"c{i}" for i in range(C)],
        physical_dims=physical_scale,
        physical_units=physical_units,
        channel_colors=[0xFFFFFF for i in range(C)],
    )
    writer.write_metadata(meta)

    # Read written result and check basics
    reader = Reader(parse_url(save_uri))
    node = list(reader())[0]
    num_levels_read = len(node.data)
    assert num_levels_read == num_levels
    for level, shape in zip(range(num_levels), expected_shapes):
        read_shape = node.data[level].shape
        assert read_shape == shape
    axes = node.metadata["axes"]
    dims = "".join([a["name"] for a in axes]).upper()
    assert dims == "TCZYX"


@array_constructor
@pytest.mark.parametrize(
    "shape, num_levels, scaling, expected_shapes",
    [
        (
            (4, 2, 2, 64, 32),  # easy, powers of two
            3,
            (1, 1, 1, 2, 2),  # downscale xy by two
            [(4, 2, 2, 64, 32), (4, 2, 2, 32, 16), (4, 2, 2, 16, 8)],
        ),
        (
            (4, 2, 2, 8, 6),
            1,  # no downscaling
            (1, 1, 1, 1, 1),
            [(4, 2, 2, 8, 6)],
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.zarr"])
def test_write_ome_zarr_iterative(
    array_constructor: Callable,
    filename: str,
    shape: DimTuple,
    num_levels: int,
    scaling: Tuple[float, float, float, float, float],
    expected_shapes: List[DimTuple],
    tmp_path: pathlib.Path,
) -> None:
    # TCZYX order, downsampling x and y only
    im = array_constructor(shape, dtype=np.uint8)
    C = shape[1]

    shapes = compute_level_shapes(shape, scaling, num_levels)
    chunk_sizes = compute_level_chunk_sizes_zslice(shapes)

    # Create an OmeZarrWriter object
    writer = OmeZarrWriter()

    # Initialize the store. Use s3 url or local directory path!
    save_uri = tmp_path / filename
    writer.init_store(str(save_uri), shapes, chunk_sizes, im.dtype)

    # Write the image iteratively as if we only have one timepoint at a time
    for t in range(shape[0]):
        t4d = im[t]
        t5d = np.expand_dims(t4d, axis=0)
        writer.write_t_batches_array(t5d, channels=[], tbatch=1, toffset=t)

    # TODO: get this from source image
    physical_scale = {
        "c": 1.0,  # default value for channel
        "t": 1.0,
        "z": 1.0,
        "y": 1.0,
        "x": 1.0,
    }
    physical_units = {
        "x": "micrometer",
        "y": "micrometer",
        "z": "micrometer",
        "t": "minute",
    }
    meta = writer.generate_metadata(
        image_name="TEST",
        channel_names=[f"c{i}" for i in range(C)],
        physical_dims=physical_scale,
        physical_units=physical_units,
        channel_colors=[0xFFFFFF for i in range(C)],
    )
    writer.write_metadata(meta)

    # Read written result and check basics
    reader = Reader(parse_url(save_uri))
    node = list(reader())[0]
    num_levels_read = len(node.data)
    assert num_levels_read == num_levels
    for level, shape in zip(range(num_levels), expected_shapes):
        read_shape = node.data[level].shape
        assert read_shape == shape
    axes = node.metadata["axes"]
    dims = "".join([a["name"] for a in axes]).upper()
    assert dims == "TCZYX"

    # check lvl 0 data values got written in order
    for t in range(shape[0]):
        t4d = im[t]
        read_t4d = node.data[0][t]
        np.testing.assert_array_equal(t4d, read_t4d)
