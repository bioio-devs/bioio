import logging
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, List, Optional, Tuple, Union, cast

import dask.array as da
import numcodecs
import numpy as np
import skimage.transform
import zarr
from ngff_zarr.zarr_metadata import Axis, Dataset, Metadata, Scale, Translation
from zarr.storage import FsspecStore, LocalStore

from bioio import BioImage

log = logging.getLogger(__name__)

DimTuple = Tuple[int, int, int, int, int]

OME_NGFF_VERSION = "0.4"


def chunk_size_from_memory_target(
    shape: DimTuple, dtype: str, memory_target: int
) -> DimTuple:
    if len(shape) != 5:
        raise ValueError("shape must be a 5-tuple in TCZYX order")
    itemsize = np.dtype(dtype).itemsize
    chunk_size: DimTuple = (1, 1, shape[2], shape[3], shape[4])
    while prod(chunk_size) * itemsize > memory_target:
        chunk_size = cast(DimTuple, tuple(max(s // 2, 1) for s in chunk_size))
    return chunk_size


def dim_tuple_to_dict(
    dims: Union[DimTuple, Tuple[float, float, float, float, float]]
) -> dict:
    if len(dims) != 5:
        raise ValueError("dims must be a 5-tuple in TCZYX order")
    return {k: v for k, v in zip(("t", "c", "z", "y", "x"), dims)}


def resize(
    image: da.Array, output_shape: Tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    factors = np.array(output_shape) / np.array(image.shape, float)
    better_chunksize = tuple(
        np.maximum(1, np.ceil(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)

    block_output_shape = tuple(
        np.ceil(np.array(better_chunksize) * factors).astype(int)
    )

    def resize_block(image_block: da.Array, block_info: dict) -> da.Array:
        chunk_output_shape = tuple(
            np.ceil(np.array(image_block.shape) * factors).astype(int)
        )
        return skimage.transform.resize(
            image_block, chunk_output_shape, *args, **kwargs
        ).astype(image_block.dtype)

    output_slices = tuple(slice(0, d) for d in output_shape)
    output = da.map_blocks(
        resize_block, image_prepared, dtype=image.dtype, chunks=block_output_shape
    )[output_slices]
    return output.rechunk(image.chunksize).astype(image.dtype)


def _pop_metadata_optionals(metadata_dict: dict) -> dict:
    for ax in metadata_dict.get("axes", []):
        if ax.get("unit") is None:
            ax.pop("unit", None)
    if metadata_dict.get("coordinateTransformations") is None:
        metadata_dict.pop("coordinateTransformations", None)
    return metadata_dict


def build_ome(
    size_z: int,
    image_name: str,
    channel_names: List[str],
    channel_colors: List[int],
    channel_minmax: List[Tuple[float, float]],
) -> dict:
    ch = []
    for i, name in enumerate(channel_names):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": name,
                "window": {
                    "end": float(channel_minmax[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_minmax[i][0]),
                },
            }
        )
    omero = {
        "id": 1,
        "name": image_name,
        "version": OME_NGFF_VERSION,
        "channels": ch,
        "rdefs": {"defaultT": 0, "defaultZ": size_z // 2, "model": "color"},
    }
    return omero


@dataclass
class ZarrLevel:
    shape: DimTuple
    chunk_size: DimTuple
    dtype: np.dtype
    zarray: zarr.Array


def compute_level_shapes(
    lvl0shape: DimTuple, scaling: Tuple[float, float, float, float, float], nlevels: int
) -> List[DimTuple]:
    shapes = [lvl0shape]
    for _ in range(nlevels - 1):
        prev = shapes[-1]
        nextshape = cast(
            DimTuple, tuple(max(int(prev[i] / scaling[i]), 1) for i in range(5))
        )
        shapes.append(nextshape)
    return shapes


def get_scale_ratio(
    level0: Tuple[int, ...], level1: Tuple[int, ...]
) -> Tuple[float, ...]:
    return tuple(level0[i] / level1[i] for i in range(len(level0)))


def compute_level_chunk_sizes_zslice(shapes: List[DimTuple]) -> List[DimTuple]:
    chunk_sizes = [(1, 1, 1, shapes[0][3], shapes[0][4])]
    for i in range(1, len(shapes)):
        scale = get_scale_ratio(shapes[i - 1], shapes[i])
        prev = chunk_sizes[i - 1]
        new: DimTuple = (
            1,
            1,
            int(scale[4] * scale[3] * prev[2]),
            int(prev[3] / scale[3]),
            int(prev[4] / scale[4]),
        )
        chunk_sizes.append(new)
    return chunk_sizes


class OmeZarrWriter:
    """Class to write OME-Zarr files."""

    def __init__(self) -> None:
        self.output_path = ""
        self.levels: List[ZarrLevel] = []
        self.store = None
        self.root: Optional[zarr.hierarchy.Group] = None

    def init_store(
        self,
        output_path: str,
        shapes: List[DimTuple],
        chunk_sizes: List[DimTuple],
        dtype: np.dtype,
        compressor: numcodecs.abc.Codec | None = None,
    ) -> None:
        if len(shapes) != len(chunk_sizes) or not shapes:
            raise ValueError("shapes and chunk_sizes must align and be non-empty")

        self.output_path = output_path
        is_remote = output_path.startswith("s3://") or output_path.startswith("gs://")
        if is_remote:
            prefix = output_path.split("://", 1)[1]
            import fsspec

            protocol = "s3" if output_path.startswith("s3://") else "gcs"
            fs = fsspec.filesystem(protocol)
            self.store = FsspecStore(fs=fs, path=prefix)
        else:
            self.store = LocalStore(output_path)

        self.root = zarr.group(store=self.store, overwrite=True)
        self._create_levels(
            root=self.root,
            level_shapes=shapes,
            level_chunk_sizes=chunk_sizes,
            dtype=dtype,
            compressor=compressor,
        )

    def _create_levels(
        self,
        root: zarr.Group,
        level_shapes: List[DimTuple],
        level_chunk_sizes: List[DimTuple],
        dtype: np.dtype,
        compressor: numcodecs.abc.Codec | None = None,
    ) -> None:
        self.levels = []
        for i, shape in enumerate(level_shapes):
            arr = root.zeros(
                str(i),
                shape=shape,
                chunks=level_chunk_sizes[i],
                dtype=dtype,
                compressor=compressor,
            )
            self.levels.append(ZarrLevel(shape, level_chunk_sizes[i], dtype, arr))

    def _downsample_and_write_batch_t(
        self, data_tczyx: da.Array, start_t: int, end_t: int, toffset: int = 0
    ) -> None:
        dtype = data_tczyx.dtype
        for k in range(start_t, end_t):
            subset = data_tczyx[[k - start_t]]
            da.to_zarr(
                subset,
                self.levels[0].zarray,
                region=(slice(k + toffset, k + toffset + 1),),
            )
        for j in range(1, len(self.levels)):
            nextshape = (end_t - start_t,) + self.levels[j].shape[1:]
            data_tczyx = resize(data_tczyx, nextshape, order=0).astype(dtype)
            for k in range(start_t, end_t):
                subset = data_tczyx[[k - start_t]]
                da.to_zarr(
                    subset,
                    self.levels[j].zarray,
                    region=(slice(k + toffset, k + toffset + 1),),
                )
        log.info(f"Completed {start_t} to {end_t}")

    def write_t_batches(
        self,
        im: BioImage,
        channels: List[int] = [],
        tbatch: int = 4,
        debug: bool = False,
    ) -> None:
        """
        Write the image in batches of T.

        Parameters
        ----------
        im:
            The BioImage object.
        tbatch:
            The number of T to write at a time.
        """
        # loop over T in batches
        numT = im.dims.T
        if debug:
            numT = np.min([5, numT])
        log.info("Starting loop over T")
        for i in np.arange(0, numT + 1, tbatch):
            start_t = i
            end_t = min(i + tbatch, numT)
            if end_t > start_t:
                # assume start t and end t are in range (caller should guarantee this)
                ti = im.get_image_dask_data(
                    "TCZYX", T=slice(start_t, end_t), C=channels
                )
                self._downsample_and_write_batch_t(ti, start_t, end_t)
        log.info("Finished loop over T")

    def write_t_batches_image_sequence(
        self,
        paths: List[str],
        channels: List[int] = [],
        tbatch: int = 4,
        debug: bool = False,
    ) -> None:
        """
        Write the image in batches of T.

        Parameters
        ----------
        paths:
            The list of file paths, one path per T.
        tbatch:
            The number of T to write at a time.
        """
        # loop over T in batches
        numT = len(paths)
        if debug:
            numT = np.min([5, numT])
        log.info("Starting loop over T")
        for i in np.arange(0, numT + 1, tbatch):
            start_t = i
            end_t = min(i + tbatch, numT)
            if end_t > start_t:
                # read batch into dask array
                ti = []
                for j in range(start_t, end_t):
                    im = BioImage(paths[j])
                    ti.append(im.get_image_dask_data("CZYX", C=channels))
                ti = da.stack(ti, axis=0)
                self._downsample_and_write_batch_t(ti, start_t, end_t)
        log.info("Finished loop over T")

    def write_t_batches_array(
        self,
        im: Union[da.Array, np.ndarray],
        channels: List[int] = [],
        tbatch: int = 4,
        toffset: int = 0,
        debug: bool = False,
    ) -> None:
        """
        Write the image in batches of T.

        Parameters
        ----------
        im:
            An ArrayLike object. Should be 5D TCZYX.
        tbatch:
            The number of T to write at a time.
        toffset:
            The offset to start writing T from. All T in the input array will be written
        """
        # if isinstance(im, (np.ndarray)):
        #     im_da = da.from_array(im)
        # else:
        #     im_da = im
        im_da = im
        # loop over T in batches
        numT = im_da.shape[0]
        if debug:
            numT = np.min([5, numT])
        log.info("Starting loop over T")
        for i in np.arange(0, numT + 1, tbatch):
            start_t = i
            end_t = min(i + tbatch, numT)
            if end_t > start_t:
                # assume start t and end t are in range (caller should guarantee this)
                ti = im_da[start_t:end_t]
                if channels:
                    for t in range(len(ti)):
                        ti[t] = [ti[t][c] for c in channels]
                self._downsample_and_write_batch_t(
                    da.asarray(ti), start_t, end_t, toffset
                )
        log.info("Finished loop over T")

    def _get_scale_ratio(self, level: int) -> Tuple[float, float, float, float, float]:
        lvl_shape = self.levels[level].shape
        lvl0_shape = self.levels[0].shape
        return (
            lvl0_shape[0] / lvl_shape[0],
            lvl0_shape[1] / lvl_shape[1],
            lvl0_shape[2] / lvl_shape[2],
            lvl0_shape[3] / lvl_shape[3],
            lvl0_shape[4] / lvl_shape[4],
        )

    def generate_metadata(
        self,
        image_name: str,
        channel_names: List[str],
        physical_dims: dict,  # {"x":0.1, "y", 0.1, "z", 0.3, "t": 5.0}
        physical_units: dict,  # {"x":"micrometer", "y":"micrometer",
        # "z":"micrometer", "t":"minute"},
        channel_colors: Union[List[str], List[int]],
    ) -> dict:
        """
        Build a metadata dict suitable for writing to ome-zarr attrs.

        Parameters
        ----------
        image_name:
            The image name.
        channel_names:
            The channel names.
        physical_dims:
            for each physical dimension, include a scale
            factor.  E.g. {"x":0.1, "y", 0.1, "z", 0.3, "t": 5.0}
        physical_units:
            For each physical dimension, include a unit
            string. E.g. {"x":"micrometer", "y":"micrometer", "z":"micrometer",
            "t":"minute"}
        """
        dims = ("t", "c", "z", "y", "x")
        axes = []
        for dim in dims:
            unit = None
            if physical_units and dim in physical_units:
                unit = physical_units[dim]
            if dim in {"x", "y", "z"}:
                axis = Axis(name=dim, type="space", unit=unit)
            elif dim == "c":
                axis = Axis(name=dim, type="channel", unit=unit)
            elif dim == "t":
                axis = Axis(name=dim, type="time", unit=unit)
            else:
                msg = f"Dimension identifier is not valid: {dim}"
                raise KeyError(msg)
            axes.append(axis)

        datasets = []
        for index, level in enumerate(self.levels):
            path = f"{index}"
            scale = []
            level_scale = self._get_scale_ratio(index)
            level_scale_dict = dim_tuple_to_dict(level_scale)
            for dim in dims:
                phys = (
                    physical_dims[dim] * level_scale_dict[dim]
                    if dim in physical_dims and dim in level_scale_dict
                    else 1.0
                )
                scale.append(phys)
            translation = []
            for dim in dims:
                # TODO handle optional translations e.g. xy stage position,
                # start time etc
                translation.append(0.0)

            coordinateTransformations = (Scale(scale), Translation(translation))
            dataset = Dataset(
                path=path, coordinateTransformations=coordinateTransformations
            )
            datasets.append(dataset)

        metadata = Metadata(
            axes=axes,
            datasets=datasets,
            name="/",
            coordinateTransformations=None,
        )
        metadata_dict = asdict(metadata)
        metadata_dict = _pop_metadata_optionals(metadata_dict)

        # get the total shape as dict:
        shapedict = dim_tuple_to_dict(self.levels[0].shape)

        # add the omero data
        ome_json = build_ome(
            shapedict["z"] if "z" in shapedict else 1,
            image_name,
            channel_names=channel_names,  # assumes we have written all channels!
            channel_colors=channel_colors,  # type: ignore
            # TODO: Rely on user to supply the per-channel min/max.
            channel_minmax=[
                (0.0, 1.0) for i in range(shapedict["c"] if "c" in shapedict else 1)
            ],
        )

        ome_zarr_metadata = {"multiscales": [metadata_dict], "omero": ome_json}
        return ome_zarr_metadata

    def write_metadata(self, metadata: dict) -> None:
        """
        Write the metadata.

        Parameters
        ----------
        metadata:
            The metadata dict. Expected to contain a multiscales
            array and omero dict
        """
        if self.root is None:
            raise RuntimeError("`init_store()` must be called before writing metadata.")
        self.root.attrs["multiscales"] = metadata["multiscales"]
        self.root.attrs["omero"] = metadata["omero"]
