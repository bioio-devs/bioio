#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from typing import Callable, List, Optional, Tuple

import bioio_base as biob
import numpy as np
import pytest
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from bioio.writers import OmeZarrWriter

from ..conftest import array_constructor


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, expected_read_shape, expected_read_dim_order",
    [
        ((1, 2, 3, 4, 5), None, (1, 2, 3, 4, 5), "TCZYX"),
        ((1, 2, 3, 4, 5), "TCZYX", (1, 2, 3, 4, 5), "TCZYX"),
        ((2, 3, 4, 5, 6), None, (2, 3, 4, 5, 6), "TCZYX"),
        ((1, 1, 1, 1, 1), None, (1, 1, 1, 1, 1), "TCZYX"),
        ((5, 16, 16), None, (5, 16, 16), "ZYX"),
        ((5, 16, 16), "ZYX", (5, 16, 16), "ZYX"),
        ((5, 16, 16), "CYX", (5, 16, 16), "CYX"),
        ((5, 16, 16), "TYX", (5, 16, 16), "TYX"),
        pytest.param(
            (10, 5, 16, 16),
            "ZCYX",
            (10, 5, 16, 16),
            "ZCYX",
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
        ((5, 10, 16, 16), "CZYX", (5, 10, 16, 16), "CZYX"),
        ((15, 16), "YX", (15, 16), "YX"),
        pytest.param(
            (2, 3, 3),
            "AYX",
            None,
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
        ((2, 3, 3), "YXZ", (2, 3, 3), "YXZ"),
        pytest.param(
            (2, 5, 16, 16),
            "CYX",
            None,
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
        # error 6D data doesn't work yet
        pytest.param(
            (1, 2, 3, 4, 5, 3),
            None,
            None,
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.zarr"])
def test_ome_zarr_writer_dims(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    write_dim_order: Optional[str],
    expected_read_shape: Tuple[int, ...],
    expected_read_dim_order: str,
    filename: str,
    tmp_path: pathlib.Path,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = tmp_path / filename

    # Normal save
    writer = OmeZarrWriter(save_uri)
    writer.write_image(arr, "", None, None, None, dimension_order=write_dim_order)

    # Read written result and check basics
    reader = Reader(parse_url(save_uri))
    node = list(reader())[0]
    num_levels = len(node.data)
    assert num_levels == 1
    level = 0
    shape = node.data[level].shape
    assert shape == expected_read_shape
    axes = node.metadata["axes"]
    dims = "".join([a["name"] for a in axes]).upper()
    assert dims == expected_read_dim_order


@array_constructor
@pytest.mark.parametrize(
    "write_shape, num_levels, scale, expected_read_shapes, expected_read_scales",
    [
        (
            (2, 4, 8, 16, 32),
            2,
            2,
            [(2, 4, 8, 16, 32), (2, 4, 8, 8, 16), (2, 4, 8, 4, 8)],
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 4.0, 4.0],
            ],
        ),
        (
            (16, 32),
            2,
            4,
            [(16, 32), (4, 8), (1, 2)],
            [
                [1.0, 1.0],
                [4.0, 4.0],
                [16.0, 16.0],
            ],
        ),
    ],
)
@pytest.mark.parametrize("filename", ["f.zarr"])
def test_ome_zarr_writer_scaling(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    num_levels: int,
    scale: float,
    expected_read_shapes: List[Tuple[int, ...]],
    expected_read_scales: List[List[int]],
    filename: str,
    tmp_path: pathlib.Path,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = tmp_path / filename

    # Normal save
    writer = OmeZarrWriter(save_uri)
    writer.write_image(
        arr, "", None, None, None, scale_num_levels=num_levels, scale_factor=scale
    )

    # Read written result and check basics
    reader = Reader(parse_url(save_uri))
    node = list(reader())[0]
    read_num_levels = len(node.data)
    assert num_levels == read_num_levels
    print(node.metadata)
    for i in range(num_levels):
        shape = node.data[i].shape
        assert shape == expected_read_shapes[i]
        xforms = node.metadata["coordinateTransformations"][i]
        assert len(xforms) == 1
        assert xforms[0]["type"] == "scale"
        assert xforms[0]["scale"] == expected_read_scales[i]


@array_constructor
@pytest.mark.parametrize(
    "write_shape, chunk_dims, num_levels, expected_read_shapes",
    [
        (
            (2, 4, 8, 16, 32),
            (1, 1, 2, 16, 16),
            2,
            [(2, 4, 8, 16, 32), (2, 4, 8, 8, 16), (2, 4, 8, 4, 8)],
        ),
        (
            (16, 32),
            (2, 4),
            2,
            [(16, 32), (8, 16), (4, 8)],
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.zarr"])
def test_ome_zarr_writer_chunks(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    chunk_dims: Tuple[int, ...],
    num_levels: int,
    filename: str,
    expected_read_shapes: List[Tuple[int, ...]],
    tmp_path: pathlib.Path,
) -> None:
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point

    baseline_save_uri = tmp_path / f"baseline_{filename}"
    save_uri = tmp_path / filename

    # Normal save
    writer = OmeZarrWriter(save_uri)
    writer.write_image(
        arr, "", None, None, None, chunk_dims=chunk_dims, scale_num_levels=num_levels
    )
    reader = Reader(parse_url(save_uri))
    node = list(reader())[0]

    # Check expected shapes
    for level in range(num_levels):
        shape = node.data[level].shape
        assert shape == expected_read_shapes[level]

    # Create baseline chunking to compare against manual.
    writer = OmeZarrWriter(baseline_save_uri)
    writer.write_image(arr, "", None, None, None, scale_num_levels=num_levels)
    reader_baseline = Reader(parse_url(baseline_save_uri))
    node_baseline = list(reader_baseline())[0]

    data = node.data[0]
    baseline_data = node_baseline.data[0]

    assert np.all(np.equal(data, baseline_data))
