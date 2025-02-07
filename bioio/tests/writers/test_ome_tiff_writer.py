#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from typing import Callable, List, Optional, Tuple, Union

import bioio_base as biob
import numpy as np
import pytest
import tifffile
from ome_types import to_xml
from ome_types.model import OME

from bioio.writers import OmeTiffWriter

from ..conftest import array_constructor


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, expected_read_shape, expected_read_dim_order",
    [
        ((5, 16, 16), None, (5, 16, 16), "ZYX"),
        ((5, 16, 16), "ZYX", (5, 16, 16), "ZYX"),
        ((5, 16, 16), "CYX", (5, 16, 16), "CYX"),
        ((10, 5, 16, 16), "ZCYX", (10, 5, 16, 16), "ZCYX"),
        ((5, 10, 16, 16), "CZYX", (5, 10, 16, 16), "CZYX"),
        ((16, 16), "YX", (16, 16), "YX"),
        pytest.param(
            (2, 3, 3),
            "AYX",
            None,
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            (2, 3, 3),
            "YXZ",
            None,
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            (2, 5, 16, 16),
            "CYX",
            None,
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
        ((1, 2, 3, 4, 5), None, (2, 3, 4, 5), "CZYX"),
        ((2, 3, 4, 5, 6), "TCZYX", (2, 3, 4, 5, 6), "TCZYX"),
        ((2, 3, 4, 5, 6), None, (2, 3, 4, 5, 6), "TCZYX"),
        ((1, 2, 3, 4, 5, 3), None, (2, 3, 4, 5, 3), "CZYXS"),
        # error 6D data doesn't work unless last dim is 3 or 4
        pytest.param(
            (1, 2, 3, 4, 5, 6),
            None,
            (1, 2, 3, 4, 5, 6),
            "TCZYXS",
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        ((5, 16, 16, 3), "ZYXS", (5, 16, 16, 3), "ZYXS"),
        ((5, 16, 16, 4), "CYXS", (5, 16, 16, 4), "CYXS"),
        ((3, 5, 16, 16, 4), "ZCYXS", (3, 5, 16, 16, 4), "ZCYXS"),
    ],
)
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_no_meta(
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
    OmeTiffWriter.save(arr, save_uri, write_dim_order)

    fs, path = biob.io.pathlike_to_fs(save_uri)
    with fs.open(path) as open_resource:
        with tifffile.TiffFile(open_resource, is_mmstack=False) as tiff:
            assert len(tiff.series) == 1
            scene = tiff.series[0]
            assert scene.shape == expected_read_shape
            assert scene.pages.axes == expected_read_dim_order


@array_constructor
@pytest.mark.parametrize(
    "shape_to_create, ome_xml, expected_shape, expected_dim_order",
    [
        # ok dims
        (
            (1, 2, 3, 4, 5),
            to_xml(OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.uint8)])),
            (2, 3, 4, 5),
            "CZYX",
        ),
        (
            (1, 2, 3, 4, 5),
            OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.uint8)]),
            (2, 3, 4, 5),
            "CZYX",
        ),
        # with RGB data:
        (
            (2, 2, 3, 4, 5, 3),
            to_xml(OmeTiffWriter.build_ome([(2, 2, 3, 4, 5, 3)], [np.dtype(np.uint8)])),
            (2, 2, 3, 4, 5, 3),
            "TCZYXS",
        ),
        (
            (2, 2, 3, 4, 5, 3),
            OmeTiffWriter.build_ome([(2, 2, 3, 4, 5, 3)], [np.dtype(np.uint8)]),
            (2, 2, 3, 4, 5, 3),
            "TCZYXS",
        ),
        # wrong dtype
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.float32)])),
            (2, 3, 4, 5),
            "CZYX",
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # wrong dtype
        pytest.param(
            (1, 2, 3, 4, 5),
            OmeTiffWriter.build_ome([(1, 2, 3, 4, 5)], [np.dtype(np.float32)]),
            (2, 3, 4, 5),
            "CZYX",
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # wrong dims
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OmeTiffWriter.build_ome([(2, 2, 3, 4, 5)], [np.dtype(np.float32)])),
            (2, 3, 4, 5),
            "CZYX",
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # wrong dims
        pytest.param(
            (1, 2, 3, 4, 5),
            OmeTiffWriter.build_ome([(2, 2, 3, 4, 5)], [np.dtype(np.float32)]),
            (2, 3, 4, 5),
            "CZYX",
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # just totally wrong but valid ome
        pytest.param(
            (1, 2, 3, 4, 5),
            to_xml(OME()),
            (2, 3, 4, 5),
            "CZYX",
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # just totally wrong but valid ome
        pytest.param(
            (1, 2, 3, 4, 5),
            OME(),
            (2, 3, 4, 5),
            "CZYX",
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # even more blatantly bad ome
        pytest.param(
            (1, 2, 3, 4, 5),
            "bad ome string",
            (2, 3, 4, 5),
            "CZYX",
            # raised from within ome-types
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_with_meta(
    array_constructor: Callable,
    shape_to_create: Tuple[int, ...],
    ome_xml: Union[str, OME, None],
    expected_shape: Tuple[int, ...],
    expected_dim_order: Tuple[str, ...],
    filename: str,
    tmp_path: pathlib.Path,
) -> None:
    # Create array
    arr = array_constructor(shape_to_create, dtype=np.uint8)

    # Construct save end point
    save_uri = tmp_path / filename

    # Normal save
    OmeTiffWriter.save(arr, save_uri, dimension_order=None, ome_xml=ome_xml)

    # Check basics
    fs, path = biob.io.pathlike_to_fs(save_uri)
    with fs.open(path) as open_resource:
        with tifffile.TiffFile(open_resource, is_mmstack=False) as tiff:
            assert len(tiff.series) == 1
            scene = tiff.series[0]
            assert scene.shape == tuple(expected_shape)
            assert scene.pages.axes == expected_dim_order


@pytest.mark.parametrize(
    "array_data, write_dim_order, read_shapes, read_dim_order",
    [
        ([np.random.rand(5, 16, 16)], None, [(5, 16, 16)], ["ZYX"]),
        (
            [np.random.rand(5, 16, 16), np.random.rand(4, 12, 12)],
            None,
            [(5, 16, 16), (4, 12, 12)],
            ["ZYX", "ZYX"],
        ),
        (
            [np.random.rand(5, 16, 16, 3), np.random.rand(4, 12, 12, 3)],
            None,
            [(5, 16, 16, 3), (4, 12, 12, 3)],
            ["CZYX", "CZYX"],
        ),
        (
            [np.random.rand(5, 16, 16, 3), np.random.rand(4, 12, 12, 3)],
            ["ZYXS", "CYXS"],
            [(5, 16, 16, 3), (4, 12, 12, 3)],
            ["ZYXS", "CYXS"],
        ),
        # spread dim_order to each image written
        (
            [np.random.rand(3, 10, 16, 16), np.random.rand(4, 12, 16, 16)],
            "CZYX",
            [(3, 10, 16, 16), (4, 12, 16, 16)],
            ["CZYX", "CZYX"],
        ),
        # different dims, rgb last
        (
            [np.random.rand(5, 16, 16), np.random.rand(4, 12, 12, 3)],
            ["ZYX", "CYXS"],
            [(5, 16, 16), (4, 12, 12, 3)],
            ["ZYX", "CYXS"],
        ),
        # different dims, rgb first
        (
            [np.random.rand(5, 16, 16, 3), np.random.rand(4, 12, 12)],
            ["ZYXS", "CYX"],
            [(5, 16, 16, 3), (4, 12, 12)],
            ["ZYXS", "CYX"],
        ),
        # two scenes but only one dimension order as list
        pytest.param(
            [np.random.rand(5, 16, 16, 3), np.random.rand(4, 12, 12)],
            ["ZYXS"],
            None,
            None,
            marks=pytest.mark.xfail(raises=biob.exceptions.ConflictingArgumentsError),
        ),
        # bad dims
        pytest.param(
            [np.random.rand(2, 3, 3)],
            "AYX",
            None,
            None,
            marks=pytest.mark.xfail(
                raises=biob.exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_multiscene(
    array_data: List[biob.types.ArrayLike],
    write_dim_order: List[Optional[str]],
    read_shapes: List[Tuple[int, ...]],
    read_dim_order: List[str],
    filename: str,
    tmp_path: pathlib.Path,
) -> None:
    # Construct save end point
    save_uri = tmp_path / filename

    # Normal save
    OmeTiffWriter.save(array_data, save_uri, write_dim_order)

    # Check basics
    fs, path = biob.io.pathlike_to_fs(save_uri)
    with fs.open(path) as open_resource:
        with tifffile.TiffFile(open_resource, is_mmstack=False) as tiff:
            assert len(tiff.series) == len(read_shapes)
            for i in range(len(tiff.series)):
                scene = tiff.series[i]
                assert scene.shape == tuple(read_shapes[i])
                assert scene.pages.axes == read_dim_order[i]


@pytest.mark.parametrize(
    "array_data, "
    "write_dim_order, "
    "pixel_size, "
    "channel_names, "
    "channel_colors, "
    "read_shapes, "
    "read_dim_order",
    [
        (
            np.random.rand(1, 2, 5, 16, 16),
            "TCZYX",
            None,
            ["C0", "C1"],
            None,
            [(2, 5, 16, 16)],
            ["CZYX"],
        ),
        (
            [np.random.rand(1, 2, 5, 16, 16), np.random.rand(1, 2, 4, 15, 15)],
            "TCZYX",
            None,
            ["C0", "C1"],
            None,
            [(2, 5, 16, 16), (2, 4, 15, 15)],
            ["CZYX", "CZYX"],
        ),
        (
            [np.random.rand(5, 16, 16)],
            None,
            [biob.types.PhysicalPixelSizes(1.0, 2.0, 3.0)],
            ["C0"],
            None,
            [(5, 16, 16)],
            ["ZYX"],
        ),
        (
            [np.random.rand(5, 16, 16)],
            None,
            [biob.types.PhysicalPixelSizes(None, 2.0, 3.0)],
            ["C0"],
            None,
            [(5, 16, 16)],
            ["ZYX"],
        ),
        (
            [np.random.rand(2, 16, 16), np.random.rand(2, 12, 12)],
            "CYX",
            [
                biob.types.PhysicalPixelSizes(1.0, 2.0, 3.0),
                biob.types.PhysicalPixelSizes(4.0, 5.0, 6.0),
            ],
            [["C0", "C1"], None],
            None,
            [(2, 16, 16), (2, 12, 12)],
            ["CYX", "CYX"],
        ),
        (
            np.random.rand(3, 16, 16),
            "CYX",
            biob.types.PhysicalPixelSizes(None, 1.0, 1.0),
            ["C0", "C1", "C2"],
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [(3, 16, 16)],
            ["CYX"],
        ),
        pytest.param(
            np.random.rand(3, 16, 16),
            "CYX",
            biob.types.PhysicalPixelSizes(None, 1.0, 1.0),
            ["C0", "C1", "C2"],
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [1, 1, 1]],
            [(3, 16, 16)],
            ["CYX"],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            [np.random.rand(3, 16, 16)],
            ["CYX"],
            [biob.types.PhysicalPixelSizes(None, 1.0, 1.0)],
            [["C0", "C1", "C2"]],
            [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]],
            [(3, 16, 16)],
            ["CYX"],
        ),
        (
            [np.random.rand(3, 16, 16)],
            ["CYX"],
            [biob.types.PhysicalPixelSizes(None, 1.0, 1.0)],
            [["C0", "C1", "C2"]],
            [None],
            [(3, 16, 16)],
            ["CYX"],
        ),
        (
            [np.random.rand(3, 16, 16), np.random.rand(3, 16, 16)],
            "CYX",
            biob.types.PhysicalPixelSizes(None, 1.0, 1.0),
            ["C0", "C1", "C2"],
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [(3, 16, 16), (3, 16, 16)],
            ["CYX", "CYX"],
        ),
        (
            [np.random.rand(3, 16, 16), np.random.rand(3, 4, 16, 16)],
            ["CYX", "CZYX"],
            [
                biob.types.PhysicalPixelSizes(None, 1.0, 1.0),
                biob.types.PhysicalPixelSizes(1.0, 1.0, 1.0),
            ],
            [["C0", "C1", "C2"], ["C4", "C5", "C6"]],
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[0, 255, 0], [0, 0, 255], [255, 0, 0]],
            ],
            [(3, 16, 16), (3, 4, 16, 16)],
            ["CYX", "CZYX"],
        ),
        (
            [np.random.rand(3, 16, 16), np.random.rand(3, 4, 16, 16)],
            ["CYX", "CZYX"],
            [
                biob.types.PhysicalPixelSizes(None, 1.0, 1.0),
                biob.types.PhysicalPixelSizes(1.0, 1.0, 1.0),
            ],
            [["C0", "C1", "C2"], ["C4", "C5", "C6"]],
            [
                None,
                [[0, 255, 0], [0, 0, 255], [255, 0, 0]],
            ],
            [(3, 16, 16), (3, 4, 16, 16)],
            ["CYX", "CZYX"],
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer_common_metadata(
    array_data: Union[biob.types.ArrayLike, List[biob.types.ArrayLike]],
    write_dim_order: Union[Optional[str], List[Optional[str]]],
    pixel_size: Union[
        biob.types.PhysicalPixelSizes, List[biob.types.PhysicalPixelSizes]
    ],
    channel_names: Union[List[str], List[Optional[List[str]]]],
    channel_colors: Union[Optional[List[List[int]]], List[Optional[List[List[int]]]]],
    read_shapes: List[Tuple[int, ...]],
    read_dim_order: List[str],
    filename: str,
    tmp_path: pathlib.Path,
) -> None:
    # Construct save end point
    save_uri = tmp_path / filename

    # Normal save
    OmeTiffWriter.save(
        array_data,
        save_uri,
        write_dim_order,
        channel_names=channel_names,
        channel_colors=channel_colors,
        physical_pixel_sizes=pixel_size,
    )

    # Check basics
    fs, path = biob.io.pathlike_to_fs(save_uri)
    with fs.open(path) as open_resource:
        with tifffile.TiffFile(open_resource, is_mmstack=False) as tiff:
            assert len(tiff.series) == len(read_shapes)
            for i in range(len(tiff.series)):
                scene = tiff.series[i]
                assert scene.shape == read_shapes[i]
                assert scene.pages.axes == read_dim_order[i]


def test_ome_tiff_writer_custom_compression(tmp_path: pathlib.Path) -> None:
    # Create array
    arr = np.random.rand(5, 16, 16)

    # Construct save end point
    save_uri = tmp_path / "e.ome.tiff"

    # Normal save
    OmeTiffWriter.save(
        arr,
        save_uri,
        tifffile_kwargs={
            "compression": "zlib",
            "compressionargs": {"level": 8},
        },
    )
