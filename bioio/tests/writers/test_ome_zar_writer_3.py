import shutil
import tempfile

import numpy as np
import pytest
import zarr
from ngff_zarr.validate import validate

from bioio.writers import OMEZarrWriter, default_axes, downsample_data


def test_downsample_data() -> None:
    data = np.arange(16, dtype=np.uint8).reshape((1, 1, 1, 4, 4))
    factors = (1, 1, 1, 2, 2)
    out = downsample_data(data, factors)
    expected = np.array([[[[[2, 4], [10, 12]]]]], dtype=np.uint8)
    assert out.shape == (1, 1, 1, 2, 2)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)


def test_default_axes() -> None:
    names = ["t", "c", "z"]
    types = ["time", "channel", "space"]
    units = [None, None, "um"]
    axes = default_axes(names, types, units)
    expected = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "um"},
    ]
    assert axes == expected


@pytest.mark.parametrize(
    "scale_factors,num_levels",
    [
        ((1, 1, 2, 2, 2), 3),
        ((1, 1, 1, 1, 1), 1),
    ],
)
def test_write_full_volume_and_metadata(
    scale_factors: tuple[int, int, int, int, int], num_levels: int
) -> None:
    # synthetic data shape
    shape = (2, 3, 4, 8, 8)
    data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    tmpdir = tempfile.mkdtemp()
    try:
        writer = OMEZarrWriter(
            store=tmpdir,
            shape=shape,
            dtype=data.dtype,
            axes_names=["t", "c", "z", "y", "x"],
            axes_types=["time", "channel", "space", "space", "space"],
            axes_units=[None, None, None, "um", "um"],
            axes_scale=[1, 1, 1, 0.5, 0.5],
            scale_factors=scale_factors,
            num_levels=num_levels,
            chunks=(1, 1, 1, 4, 4),
            shards=(1, 1, 2, 2, 2),
            channel_names=[f"c{i}" for i in range(shape[1])],
            channel_colors=["FF0000"] * shape[1],
            creator_info={"name": "test", "version": "0.1"},
        )
        writer.write_full_volume(data)
        grp = zarr.open(tmpdir, mode="r")
        # verify each level
        sf = np.array(scale_factors)
        for lvl in range(num_levels):
            arr = grp[str(lvl)]
            base = np.array(shape)
            for _ in range(lvl):
                mask = sf > 1
                base[mask] //= sf[mask]
            assert tuple(base) == arr.shape
        # verify metadata
        ome = grp.attrs["ome"]
        assert ome["version"] == "0.5"
        ms = ome["multiscales"][0]
        assert ms["name"] == "Image"
        axis_names = [a["name"] for a in ms["axes"]]
        assert axis_names == ["t", "c", "z", "y", "x"]
        assert len(ms["datasets"]) == num_levels
    finally:
        shutil.rmtree(tmpdir)


# Test compute_level_shapes matches original cases with potential early stopping
@pytest.mark.parametrize(
    "in_shape,scale_factors,num_levels,expected",
    [
        (
            (1, 1, 1, 128, 128),
            (1, 1, 1, 2, 2),
            2,
            [(1, 1, 1, 128, 128), (1, 1, 1, 64, 64)],
        ),
        (
            (1, 1, 256, 1024, 2048),
            (1, 1, 1, 2, 2),
            3,
            [(1, 1, 256, 1024, 2048), (1, 1, 256, 512, 1024), (1, 1, 256, 256, 512)],
        ),
        (
            (1, 1, 1, 4, 4),
            (1, 1, 1, 2, 2),
            5,
            [(1, 1, 1, 4, 4), (1, 1, 1, 2, 2), (1, 1, 1, 1, 1)],
        ),
    ],
)
def test_compute_level_shapes(
    in_shape: tuple[int, int, int, int, int],
    scale_factors: tuple[int, int, int, int, int],
    num_levels: int,
    expected: list[tuple[int, int, int, int, int]],
) -> None:
    writer = OMEZarrWriter(
        store=tempfile.mkdtemp(),
        shape=in_shape,
        dtype=np.uint8,
        axes_names=["t", "c", "z", "y", "x"],
        axes_types=["time", "channel", "space", "space", "space"],
        axes_units=[None, None, None, None, None],
        axes_scale=[1, 1, 1, 1, 1],
        scale_factors=scale_factors,
        num_levels=num_levels,
    )
    out = writer._compute_levels(num_levels)
    assert out == expected


def test_suggest_chunks() -> None:
    shape = (1, 1, 1, 5000, 5000)
    writer = OMEZarrWriter(store=tempfile.mkdtemp(), shape=shape, dtype=np.uint32)
    ck = writer._suggest_chunks(shape)
    assert ck == (1, 1, 1, 4096, 4096)


def test_sharding_parameter() -> None:
    shape = (1, 1, 1, 4, 4)
    requested_shards = (1, 1, 2, 2, 2)
    chunks = (1, 1, 1, 2, 2)
    writer = OMEZarrWriter(
        store=tempfile.mkdtemp(),
        shape=shape,
        dtype=np.uint8,
        chunks=chunks,
        shards=requested_shards,
    )
    # expected shards: clamp per-level to each level_shapes
    expected_shards = []
    expected_chunks = []
    for ls in writer.level_shapes:
        expected_shards.append(
            tuple(min(s, dim) for s, dim in zip(requested_shards, ls))
        )
        expected_chunks.append(tuple(min(c, dim) for c, dim in zip(chunks, ls)))
    assert writer.shards == expected_shards
    assert writer.chunks == expected_chunks


def test_ome_ngff_metadata_validation() -> None:
    # Validate OME-Zarr metadata against NGFF schema v0.5
    shape = (1, 1, 1, 4, 4)
    data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    tmpdir = tempfile.mkdtemp()
    try:
        writer = OMEZarrWriter(store=tmpdir, shape=shape, dtype=data.dtype)
        writer.write_full_volume(data)
        grp = zarr.open(tmpdir, mode="r")
        ome_meta = grp.attrs.asdict()
        validate(ome_meta, version="0.5", model="image", strict=False)
    finally:
        shutil.rmtree(tmpdir)
