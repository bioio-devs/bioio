import logging
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, List, Tuple, Union

import dask.array as da
import numcodecs
import numpy as np
import skimage.transform
import zarr
from ngff_zarr.zarr_metadata import Axis, Dataset, Metadata, Scale, Translation
from zarr.storage import DirectoryStore, FSStore, default_compressor

from bioio import BioImage

log = logging.getLogger(__name__)

DimTuple = Tuple[int, int, int, int, int]

OME_NGFF_VERSION = "0.4"


def chunk_size_from_memory_target(
    shape: DimTuple, dtype: str, memory_target: int
) -> DimTuple:
    """
    Calculate chunk size from memory target in bytes.  The chunk size will be
    determined by considering a single T and C, and subdividing the remaining
    dims by 2 until the chunk fits within the size target.

    Parameters
    ----------
    shape:
        Shape of the array. Assumes a 5d TCZYX array.
    dtype:
        Data type of the array.
    memory_target:
        Memory target in bytes.

    Returns
    -------
    Chunk size tuple.
    """
    if len(shape) != 5:
        raise ValueError("shape must be a 5-tuple in TCZYX order")

    itemsize = np.dtype(dtype).itemsize
    # let's start by just mandating that chunks have to be no more than
    # 1 T and 1 C
    chunk_size = (1, 1, shape[2], shape[3], shape[4])
    while prod(chunk_size) * itemsize > memory_target:
        # chop every dim in half until they get down to 1
        chunk_size = (
            max(chunk_size[0] // 2, 1),
            max(chunk_size[1] // 2, 1),
            max(chunk_size[2] // 2, 1),
            max(chunk_size[3] // 2, 1),
            max(chunk_size[4] // 2, 1),
        )
    return chunk_size


def dim_tuple_to_dict(
    dims: Union[DimTuple, Tuple[float, float, float, float, float]]
) -> dict:
    if len(dims) != 5:
        raise ValueError("dims must be a 5-tuple in TCZYX order")
    return {"t": dims[0], "c": dims[1], "z": dims[2], "y": dims[3], "x": dims[4]}


def resize(
    image: da.Array, output_shape: Tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    """
    Wrapped copy of "skimage.transform.resize"
    Resize image to match a certain size.

    Parameters
    ----------
    image: :class:`dask.array`
        The dask array to resize
    output_shape: tuple
        The shape of the resize array
    *args: list
        Arguments of skimage.transform.resize
    **kwargs: dict
        Keyword arguments of skimage.transform.resize

    Returns
    -------
    Resized image.
    """
    factors = np.array(output_shape) / np.array(image.shape).astype(float)
    # Rechunk the input blocks so that the factors achieve an output
    # blocks size of full numbers.
    better_chunksize = tuple(
        np.maximum(1, np.round(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)

    # If E.g. we resize image from 6675 by 0.5 to 3337, factor is 0.49992509 so each
    # chunk of size e.g. 1000 will resize to 499. When assumbled into a new array, the
    # array will now be of size 3331 instead of 3337 because each of 6 chunks was
    # smaller by 1. When we compute() this, dask will read 6 chunks of 1000 and expect
    # last chunk to be 337 but instead it will only be 331.
    # So we use ceil() here (and in resize_block) to round 499.925 up to chunk of 500
    block_output_shape = tuple(
        np.ceil(np.array(better_chunksize) * factors).astype(int)
    )

    # Map overlap
    def resize_block(image_block: da.Array, block_info: dict) -> da.Array:
        # if the input block is smaller than a 'regular' chunk (e.g. edge of image)
        # we need to calculate target size for each chunk...
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
    for ax in metadata_dict["axes"]:
        if ax["unit"] is None:
            ax.pop("unit")

    if metadata_dict["coordinateTransformations"] is None:
        metadata_dict.pop("coordinateTransformations")

    return metadata_dict


def build_ome(
    size_z: int,
    image_name: str,
    channel_names: List[str],
    channel_colors: List[int],
    channel_minmax: List[Tuple[float, float]],
) -> dict:
    """
    Create the omero metadata for an OME zarr image

    Parameters
    ----------
    size_z:
        Number of z planes
    image_name:
        The name of the image
    channel_names:
        The names for each channel.  Must be of correct length!
    channel_colors:
        List of all channel colors
    channel_minmax:
        List of all (min, max) pairs of channel intensities

    Returns
    -------
    Dict
        An "omero" metadata object suitable for writing to ome-zarr
    """
    ch = []
    for i in range(len(channel_names)):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": channel_names[i],
                "window": {
                    "end": float(channel_minmax[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_minmax[i][0]),
                },
            }
        )

    omero = {
        "id": 1,  # ID in OMERO
        "name": image_name,  # Name as shown in the UI
        "version": OME_NGFF_VERSION,  # Current version
        "channels": ch,
        "rdefs": {
            "defaultT": 0,  # First timepoint to show the user
            "defaultZ": size_z // 2,  # First Z section to show the user
            "model": "color",  # "color" or "greyscale"
        },
        # TODO: can we add more metadata here?
        # Must check with the ngff spec.
        # # from here down this is all extra and not part of the ome-zarr spec
        # "meta": {
        #     "projectDescription": "20+ lines of gene edited cells etc",
        #     "datasetName": "aics_hipsc_v2020.1",
        #     "projectId": 2,
        #     "imageDescription": "foo bar",
        #     "imageTimestamp": 1277977808.0,
        #     "imageId": 12,
        #     "imageAuthor": "danielt",
        #     "imageName": "AICS-12_143.ome.tif",
        #     "datasetDescription": "variance dataset after QC",
        #     "projectName": "aics cell variance project",
        #     "datasetId": 3
        # },
    }
    return omero


@dataclass
class ZarrLevel:
    shape: DimTuple
    chunk_size: DimTuple
    dtype: np.dtype
    zarray: zarr.core.Array


def compute_level_shapes(
    lvl0shape: DimTuple, scaling: Tuple[float, float, float, float, float], nlevels: int
) -> List[DimTuple]:
    """
    Calculate all multiresolution level shapes by repeatedly scaling.
    Minimum dimension size will always be 1.
    This will always return nlevels even if the levels become unreducible and
    have to repeat.

    Parameters
    ----------
    lvl0shape:
        Shape of the array. Assumes a 5d TCZYX tuple.
    scaling:
        Amount to scale each dimension by. Dims will be DIVIDED by
        these values.
    nlevels:
        Number of levels to return. The first level is the
        original lvl0shape.

    Returns
    -------
    List of shapes of all nlevels.
    """
    shapes = [lvl0shape]
    for i in range(nlevels - 1):
        nextshape = (
            max(int(shapes[i][0] / scaling[0]), 1),
            max(int(shapes[i][1] / scaling[1]), 1),
            max(int(shapes[i][2] / scaling[2]), 1),
            max(int(shapes[i][3] / scaling[3]), 1),
            max(int(shapes[i][4] / scaling[4]), 1),
        )
        shapes.append(nextshape)
    return shapes


def get_scale_ratio(
    level0: Tuple[int, ...], level1: Tuple[int, ...]
) -> Tuple[float, ...]:
    return (
        level0[0] / level1[0],
        level0[1] / level1[1],
        level0[2] / level1[2],
        level0[3] / level1[3],
        level0[4] / level1[4],
    )


def compute_level_chunk_sizes_zslice(shapes: List[DimTuple]) -> List[DimTuple]:
    """
    Convenience function to calculate chunk sizes for each of the input level
    shapes assuming that the shapes are TCZYX and we want the chunking to be
    per Z slice.  This code also assumes we are only downsampling in XY and
    leaving TCZ alone.  For many of our microscopy images so far, we have much
    more resolution in XY than in Z so this is a reasonable assumption.

    The first shape returned will always be (1,1,1,shapes[0][3],shapes[0][4])
    and the following will be a scaled number of slices scaled by the same
    factor as the successive shapes.

    This is an attempt to keep the total size of chunks the same across all
    levels, by increasing the number of slices for downsampled levels.
    This is making a basic assumption that each of the shapes is a downsampled
    version of the previous shape.

    For example, in a typical case, if the second level is scaled down by 1/2
    in X and Y, then the second chunk size will have 4x the number of slices.

    Parameters
    ----------
    shapes:
        List of all multiresolution level shapes

    Returns
    -------
    List of chunk sizes for per-slice chunking
    """
    # assumes TCZYX order
    shape0 = shapes[0]
    chunk_sizes = []
    # assume starting with single slice
    chunk_sizes.append((1, 1, 1, shape0[3], shape0[4]))
    for i in range(1, len(shapes)):
        last_chunk_size = chunk_sizes[i - 1]
        scale = get_scale_ratio(shapes[i - 1], shapes[i])
        # assumes that scale ratio for TCZ is 1 and only downsampled in xy.
        chunk_sizes.append(
            (
                1,
                1,
                (int(scale[4] * scale[3] * last_chunk_size[2])),
                int(last_chunk_size[3] / scale[3]),
                int(last_chunk_size[4] / scale[4]),
            )
        )
    return chunk_sizes


class OmeZarrWriter:
    """
    Class to write OME-Zarr files.
    Example usage:

    .. code-block:: python

            from ome_zarr_writer import
                OmeZarrWriter,
                compute_level_shapes,
                compute_level_chunk_sizes_zslice

            # We need to compute the shapes and chunk sizes for each
            # desired multiresolution level.
            shapes = compute_level_shapes(input_shape, scaling, num_levels)
            chunk_sizes = compute_level_chunk_sizes_zslice(shapes)

            # Create an OmeZarrWriter object
            writer = OmeZarrWriter()

            # Initialize the store. Use s3 url or local directory path!
            writer.init_store(str(save_uri), shapes, chunk_sizes, im.dtype)

            # Write the image.
            # This will compute downsampled levels on the fly.
            # Adjust t batch size based on dask compute capacity.
            writer.write_t_batches_array(im, tbatch=4)

            # Generate a metadata dict and write it to the zarr.
            meta = writer.generate_metadata(
                image_name="my_image_name",
                channel_names=my_channel_names,
                physical_dims=physical_scale,
                physical_units=physical_units,
                channel_colors=my_channel_colors,
            )
            writer.write_metadata(meta)
    """

    def __init__(self) -> None:
        self.output_path: str = ""
        self.levels: List[ZarrLevel] = []
        self.store: zarr.Store = None
        self.root: zarr.hierarchy.Group = None

    def init_store(
        self,
        output_path: str,
        shapes: List[DimTuple],
        chunk_sizes: List[DimTuple],
        dtype: np.dtype,
        compressor: numcodecs.abc.Codec = default_compressor,
    ) -> None:
        """
        Initialize the store.

        Parameters
        ----------
        output_path:
            The output path. If it begins with "s3://" or "gs://", it is assumed
            to be a remote store. Credentials required to be provided externally.
        shapes:
            The shapes of the levels.
        chunk_sizes:
            The chunk sizes of the levels.
        dtype:
            The data type.
        """
        if len(shapes) != len(chunk_sizes) or len(shapes) < 1 or len(chunk_sizes) < 1:
            raise ValueError(
                "shapes and chunk_sizes must have the same length.  This is "
                "the number of multiresolution levels."
            )

        self.output_path = output_path
        # assumes authentication/permission for writes
        is_remote = output_path.startswith("s3://") or output_path.startswith("gs://")
        if is_remote:
            self.store = FSStore(url=output_path, dimension_separator="/")
        else:
            self.store = DirectoryStore(output_path, dimension_separator="/")
        # create a group with all the levels
        self.root = zarr.group(store=self.store, overwrite=True)
        # pre-create all levels here?
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
        compressor: numcodecs.abc.Codec = default_compressor,
    ) -> None:
        self.levels = []
        for i in range(len(level_shapes)):
            lvl = (
                root.zeros(
                    str(i),
                    shape=level_shapes[i],
                    chunks=level_chunk_sizes[i],
                    dtype=dtype,
                    compressor=compressor,
                )
                if root is not None
                else None
            )
            level = ZarrLevel(level_shapes[i], level_chunk_sizes[i], dtype, lvl)
            self.levels.append(level)

    def _downsample_and_write_batch_t(
        self, data_tczyx: da.Array, start_t: int, end_t: int, toffset: int = 0
    ) -> None:
        dtype = data_tczyx.dtype
        if len(data_tczyx.shape) != 5:
            raise ValueError("data_tczyx must be 5D")
        if len(data_tczyx) != end_t - start_t:
            raise ValueError("data_tczyx must have the same T length as end_t-start_t")

        # write level 0 first
        for k in range(start_t, end_t):
            subset = data_tczyx[[k - start_t]]
            da.to_zarr(
                subset,
                self.levels[0].zarray,
                region=(slice(k + toffset, k + toffset + 1),),
            )

        # downsample to next level then write
        for j in range(1, len(self.levels)):
            # downsample to next level
            nextshape = (end_t - start_t,) + self.levels[j].shape[1:]
            data_tczyx = resize(data_tczyx, nextshape, order=0)
            data_tczyx = data_tczyx.astype(dtype)

            # write ti to zarr
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
            coordinateTransformations = [Scale(scale), Translation(translation)]
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
        self.root.attrs["multiscales"] = metadata["multiscales"]
        self.root.attrs["omero"] = metadata["omero"]
