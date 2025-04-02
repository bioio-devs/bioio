#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args

import bioio_base as biob
import dask.array as da
import numpy as np
import xarray as xr
from bioio_base.types import MetaArrayLike
from ome_types import OME

from .ome_utils import generate_ome_channel_id
from .plugins import PluginEntry, check_type, get_array_like_plugin, get_plugins

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class BioImage(biob.image_container.ImageContainer):
    """
    BioImage takes microscopy image data types (files or arrays) of varying
    dimensions ("ZYX", "TCZYX", "CYX") and reads them as consistent 5D "TCZYX"
    ("Time-Channel-Z-Y-X") ordered array(s). The data and metadata are lazy
    loaded and can be accessed as needed.

    Parameters
    ----------
    image: biob.types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    reader: Optional[Type[Reader]]
        The Reader class to specifically use for reading the provided image.
        Default: None (find matching reader)
    reconstruct_mosaic: bool
        Boolean for setting that data for this object to the reconstructed / stitched
        mosaic image.
        Default: True (reconstruct the mosaic image from tiles)
        Notes: If True and image is a mosaic, data will be fully reconstructed and
        stitched array.
        If True and base reader doesn't support tile stitching, data won't be stitched
        and instead will have an `M` dimension for tiles.
        If False and image is a mosaic, data won't be stitched and instead will have an
        `M` dimension for tiles.
        If image is not a mosaic, data won't be stitched or have an `M` dimension for
        tiles.
    use_plugin_cache: bool default False
        Boolean for setting whether to use a plugin of the installed caches rather than
        checking for installed plugins on each `BioImage` instance init.
        If True, will use the cache of installed plugins discovered last `BioImage`
        init.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    kwargs: Any
        Extra keyword arguments that will be passed down to the reader subclass.

    Examples
    --------
    Initialize an image then read the file and return specified slices as a numpy
    array.

    >>> img = BioImage("my_file.tiff")
    ... zstack_t8 = img.get_image_data("ZYX", T=8, C=0)

    Initialize an image, construct a delayed dask array for certain slices, then
    read only the specified chunk of data.

    >>> img = BioImage("my_file.czi")
    ... zstack_t8 = img.get_image_dask_data("ZYX", T=8, C=0)
    ... zstack_t8_data = zstack_t8.compute()

    Initialize an image with a dask or numpy array.

    >>> data = np.random.rand(100, 100)
    ... img = BioImage(data)

    Initialize an image from S3 with s3fs.

    >>> img = BioImage("s3://my_bucket/my_file.tiff")

    Initialize an image and pass arguments to the reader using kwargs.

    >>> img = BioImage("my_file.czi", chunk_dims=["T", "Y", "X"])

    Initialize an image, change scene, read data to numpy.

    >>> img = BioImage("my_many_scene.czi")
    ... img.set_scene("Image:3")
    ... img.data

    Initialize an image with a specific reader. This is useful if you know the file
    type in advance or would like to skip a few of the file format checks we do
    internally. Useful when reading from remote sources to reduce network round trips.

    >>> img = BioImage("malformed_metadata.ome.tiff", reader=readers.TiffReader)

    Data for a mosaic file is returned pre-stitched (if the base reader supports it).

    >>> img = BioImage("big_mosaic.czi")
    ... img.dims  # <Dimensions [T: 40, C: 3, Z: 1, Y: 30000, X: 45000]>

    Data for mosaic file can be explicitly returned as tiles.
    This is the same data as a reconstructed mosaic except that the tiles are
    stored in their own dimension (M).

    >>> img = BioImage("big_mosaic.czi", reconstruct_mosaic=False)
    ... img.dims  # <Dimensions [M: 150, T: 40, C: 3, Z: 1, Y: 200, X: 300]>

    Data is mosaic file but reader doesn't support tile stitching.

    >>> img = BioImage("unsupported_mosaic.ext")
    ... img.dims  # <Dimensions [M: 100, T: 1, C: 2, Z: 1, Y: 400, X: 400]>

    Notes
    -----
    If your image is made up of mosaic tiles, data and dimension information returned
    from this object will be from the tiles already stitched together.

    If you do not want the image pre-stitched together, you can use the base reader
    by either instantiating the reader independently or using the `.reader` property.
    """

    @staticmethod
    def determine_plugin(
        image: biob.types.ImageLike,
        fs_kwargs: Dict[str, Any] = {},
        use_plugin_cache: bool = False,
        **kwargs: Any,
    ) -> PluginEntry:
        """
        Determine the appropriate plugin to read a given image.

        This function identifies the most suitable plugin to read the provided image
        based on its type or file extension. It leverages the installed plugins for
        `bioio`, each of which supports a subset of image formats. If a suitable
        plugin is found, it is returned; otherwise, an error is raised.

        Parameters
        ----------
        image : biob.types.ImageLike
            The image to be read. This can be a file path (str or Path) or
            an array-like object (e.g., numpy array).
        fs_kwargs : Dict[str, Any], optional
            Additional keyword arguments to be passed to the file system handler.
        use_plugin_cache : bool, optional
            Whether to use a cached version of the plugin mapping, by default False.
        **kwargs : Any
            Additional keyword arguments for plugin-specific configurations.

        Returns
        -------
        PluginEntry
            A `PluginEntry` NamedTuple which is a wrapper of release information and
            reader references for an individual plugin.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            Raised if no suitable reader plugin can be found for the provided image.

        Notes
        -----
        This function performs the following steps:
        1. Fetches an updated mapping of available plugins,
           optionally using a cached version.
        2. If the `image` is a file path (str or Path), it checks for a matching
           plugin based on the file extension.
        3. If the `image` is an array-like object, it attempts to use the
           built-in `ArrayLikeReader`.
        4. If no suitable plugin is found, raises an `UnsupportedFileFormatError`.

        Examples
        --------
        To determine the appropriate plugin for a given image file:

        >>> image_path = "example_image.tif"
        >>> plugin = determine_plugin(image_path)
        >>> print(plugin)

        To determine the appropriate plugin for an array-like image:

        >>> import numpy as np
        >>> image_array = np.random.random((5, 5, 5))
        >>> plugin = determine_plugin(image_array)
        >>> print(plugin)

        Implementation Details
        ----------------------
        - The function first converts the image to a string representation.
        - If the image is a file path, it verifies the path and checks the file
          extension against the known plugins.
        - For each matching plugin, it tries to instantiate a reader and checks
          if it supports the image.
        - If the image is array-like, it uses a built-in reader designed for
          such objects.
        - Detailed logging is provided for troubleshooting purposes.
        """
        # Fetch updated mapping of plugins
        plugins_by_ext = get_plugins(use_cache=use_plugin_cache)

        # Try reader detection based off of file path extension
        image_str = str(type(image))
        if isinstance(image, (str, Path)):
            image_str = str(image)
            # we do not enforce_exists because it's possible
            # the path is a directory on a HTTP filesystem
            # which is impossible to check for existence
            _, path = biob.io.pathlike_to_fs(
                image, enforce_exists=False, fs_kwargs=fs_kwargs
            )

            # Check for extension in plugins_by_ext
            for format_ext, plugins in plugins_by_ext.items():
                if path.lower().endswith(format_ext):
                    for plugin in plugins:
                        ReaderClass = plugin.metadata.get_reader()
                        try:
                            if ReaderClass.is_supported_image(
                                image,
                                fs_kwargs=fs_kwargs,
                            ):
                                return plugin

                        except FileNotFoundError as fe:
                            raise fe
                        except Exception as e:
                            log.warning(
                                f"Attempted file ({path}) load with "
                                f"reader: {ReaderClass} failed with error: {e}"
                            )

        # Use built-in ArrayLikeReader if type MetaArrayLike
        elif isinstance(image, get_args(MetaArrayLike) + (list,)):
            return get_array_like_plugin()

        # If we haven't hit anything yet, we likely don't support this file / object
        # with the current plugins installed
        raise biob.exceptions.UnsupportedFileFormatError(
            "BioImage",
            image_str,
            msg_extra=(
                "You may need to install an extra format dependency. "
                "See our list of known plugins in the bioio README here: "
                "https://github.com/bioio-devs/bioio for a list of known plugins. "
                "You can also call the 'bioio.plugins.dump_plugins()' method to "
                "report information about currently installed plugins or the "
                "'bioio.plugin_feasibility_report(image)' method to check if a "
                "specific image can be handled by the available plugins."
            ),
        )

    @staticmethod
    def _get_reader(
        image: biob.types.ImageLike,
        reader: biob.reader.Reader,
        use_plugin_cache: bool,
        fs_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[biob.reader.Reader, Optional[PluginEntry]]:
        """
        Initializes and returns the reader (and plugin if relevant) for the provided
        image based on provided args and/or the available bioio supported plugins
        """
        if reader is not None:
            # Check specific reader image types in a situation where a specified reader
            # only supports some of the ImageLike types.
            if not check_type(image, reader):
                raise biob.exceptions.UnsupportedFileFormatError(
                    reader.__name__, str(type(image))
                )

            return reader(image, fs_kwargs=fs_kwargs, **kwargs), None

        # Determine reader class based on available plugins
        plugin = BioImage.determine_plugin(
            image, fs_kwargs=fs_kwargs, use_plugin_cache=use_plugin_cache, **kwargs
        )
        ReaderClass = plugin.metadata.get_reader()
        return ReaderClass(image, fs_kwargs=fs_kwargs, **kwargs), plugin

    def __init__(
        self,
        image: biob.types.ImageLike,
        reader: Optional[Type[biob.reader.Reader]] = None,
        reconstruct_mosaic: bool = True,
        use_plugin_cache: bool = False,
        fs_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        try:
            self._reader, self._plugin = self._get_reader(
                image,
                reader,
                use_plugin_cache,
                fs_kwargs,
                **kwargs,
            )
        except biob.exceptions.UnsupportedFileFormatError:
            # When reading from S3 if we failed trying to read
            # try reading as an anonymous user otherwise re-raise
            # the error
            if not str(image).startswith("s3://"):
                raise

            self._reader, self._plugin = self._get_reader(
                image,
                reader,
                use_plugin_cache,
                fs_kwargs | {"anon": True},
                **kwargs,
            )

        # Store delayed modifiers
        self._reconstruct_mosaic = reconstruct_mosaic

        # Lazy load data from reader and reformat to standard dimensions
        self._xarray_dask_data: Optional[xr.DataArray] = None
        self._xarray_data: Optional[xr.DataArray] = None
        self._dims: Optional[biob.dimensions.Dimensions] = None

    @property
    def reader(self) -> biob.reader.Reader:
        """
        Returns
        -------
        reader: Reader
            The object created to read the image file type.
            The intent is that if the BioImage class doesn't provide a raw enough
            interface then the base class can be used directly.
        """
        return self._reader

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        Returns
        -------
        scenes: Tuple[str, ...]
            A tuple of valid scene ids in the file.

        Notes
        -----
        Scene IDs are strings - not a range of integers.

        When iterating over scenes please use:

        >>> for id in image.scenes

        and not:

        >>> for i in range(len(image.scenes))
        """
        return self.reader.scenes

    @property
    def current_scene(self) -> str:
        """
        Returns
        -------
        scene: str
            The current operating scene.
        """
        return self.reader.current_scene

    @property
    def current_scene_index(self) -> int:
        """
        Returns
        -------
        scene_index: int
            The current operating scene index in the file.
        """
        return self.scenes.index(self.current_scene)

    def set_scene(self, scene_id: Union[str, int]) -> None:
        """
        Set the operating scene.

        Parameters
        ----------
        scene_id: Union[str, int]
            The scene id (if string) or scene index (if integer)
            to set as the operating scene.

        Raises
        ------
        IndexError
            The provided scene id or index is not found in the available scene id list.
        TypeError
            The provided value wasn't a string (scene id) or integer (scene index).
        """
        # Update current scene on the base Reader
        # This clears the base Reader's cache
        self.reader.set_scene(scene_id)

        # Reset the data stored in the BioImage object
        self._xarray_dask_data = None
        self._xarray_data = None
        self._dims = None

    def _transform_data_array_to_bioio_image_standard(
        self,
        arr: xr.DataArray,
    ) -> xr.DataArray:
        # Determine if we need to add optionally-standard dims
        if (
            biob.dimensions.DimensionNames.Samples in arr.dims
            and biob.dimensions.DimensionNames.MosaicTile in arr.dims
        ):
            return_dims = (
                biob.dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES_AND_SAMPLES
            )
        elif biob.dimensions.DimensionNames.Samples in arr.dims:
            return_dims = biob.dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES
        elif biob.dimensions.DimensionNames.MosaicTile in arr.dims:
            return_dims = biob.dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES
        else:
            return_dims = biob.dimensions.DEFAULT_DIMENSION_ORDER

        # Pull the data with the appropriate dimensions
        data = biob.transforms.reshape_data(
            data=arr.data,
            given_dims="".join(arr.dims),
            return_dims=return_dims,
        )

        # Pull coordinate planes
        coords: Dict[str, Any] = {}
        for d in return_dims:
            if d in arr.coords:
                coords[d] = arr.coords[d]

        # Add channel coordinate plane because it is required in BioImage
        if biob.dimensions.DimensionNames.Channel not in coords:
            coords[biob.dimensions.DimensionNames.Channel] = [
                generate_ome_channel_id(
                    image_id=self.current_scene,
                    channel_id=0,
                )
            ]

        return xr.DataArray(
            data,
            dims=tuple([d for d in return_dims]),
            coords=coords,
            attrs=arr.attrs,
        )

    @property
    def resolution_levels(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        resolution_levels: Tuple[int, ...]
            A tuple of valid resolution levels in the file.
        """
        return self.reader.resolution_levels

    @property
    def current_resolution_level(self) -> int:
        """
        Returns
        -------
        current_resolution_level: int
            The currently selected resolution level.
        """
        return self.reader.current_resolution_level

    def set_resolution_level(self, resolution_level: int) -> None:
        """
        Set the operating resolution level.

        Parameters
        ----------
        resolution_level: int
            The selected resolution level to set as current.

        Raises
        ------
        IndexError
            The provided level is not found in the available list of resolution levels.
        """
        if resolution_level not in self.resolution_levels:
            raise IndexError(
                f"Resolution level {resolution_level} not found in available "
                f"resolution levels {self.resolution_levels}"
            )
        if resolution_level == self.current_resolution_level:
            return

        self.reader.set_resolution_level(resolution_level)
        # Reset the data stored in the BioImage object
        self._xarray_dask_data = None
        self._xarray_data = None
        self._dims = None

    @property
    def resolution_level_dims(self) -> Dict[int, Tuple[int, ...]]:
        """
        Returns
        -------
        resolution_level_dims: Dict[int, Tuple[int, ...]]
            resolution level dictionary of shapes.
        """
        return self.reader.resolution_level_dims

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_dask_data: xr.DataArray
            The delayed image and metadata as an annotated data array.

        Notes
        -----
        If the image contains mosaic tiles, data is returned already stitched together.
        """
        if self._xarray_dask_data is None:
            if (
                # Does the user want to get stitched mosaic
                self._reconstruct_mosaic
                # Does the data have a tile dim
                and biob.dimensions.DimensionNames.MosaicTile in self.reader.dims.order
            ):
                try:
                    self._xarray_dask_data = (
                        self._transform_data_array_to_bioio_image_standard(
                            self.reader.mosaic_xarray_dask_data
                        )
                    )

                # Catch reader does not support tile stitching
                except NotImplementedError:
                    self._xarray_dask_data = (
                        self._transform_data_array_to_bioio_image_standard(
                            self.reader.xarray_dask_data
                        )
                    )

            else:
                self._xarray_dask_data = (
                    self._transform_data_array_to_bioio_image_standard(
                        self.reader.xarray_dask_data
                    )
                )

        return self._xarray_dask_data

    @property
    def xarray_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_data: xr.DataArray
            The fully read image and metadata as an annotated data array.

        Notes
        -----
        If the image contains mosaic tiles, data is returned already stitched together.
        Recommended to use `xarray_dask_data` for large mosaic images.
        """
        if self._xarray_data is None:
            if (
                # Does the user want to get stitched mosaic
                self._reconstruct_mosaic
                # Does the data have a tile dim
                and biob.dimensions.DimensionNames.MosaicTile in self.reader.dims.order
            ):
                try:
                    self._xarray_data = (
                        self._transform_data_array_to_bioio_image_standard(
                            self.reader.mosaic_xarray_data
                        )
                    )

                # Catch reader does not support tile stitching
                except NotImplementedError:
                    self._xarray_data = (
                        self._transform_data_array_to_bioio_image_standard(
                            self.reader.xarray_data
                        )
                    )

            else:
                self._xarray_data = self._transform_data_array_to_bioio_image_standard(
                    self.reader.xarray_data
                )

            # Remake the delayed xarray dataarray object using a rechunked dask array
            # from the just retrieved in-memory xarray dataarray
            self._xarray_dask_data = xr.DataArray(
                da.from_array(self._xarray_data.data),
                dims=self._xarray_data.dims,
                coords=self._xarray_data.coords,
                attrs=self._xarray_data.attrs,
            )

        return self._xarray_data

    @property
    def dask_data(self) -> da.Array:
        """
        Returns
        -------
        dask_data: da.Array
            The image as a dask array with standard dimension ordering.

        Notes
        -----
        If the image contains mosaic tiles, data is returned already stitched together.
        """
        return self.xarray_dask_data.data

    @property
    def data(self) -> np.ndarray:
        """
        Returns
        -------
        data: np.ndarray
            The image as a numpy array with standard dimension ordering.

        Notes
        -----
        If the image contains mosaic tiles, data is returned already stitched together.
        Recommended to use `dask_data` for large mosaic images.
        """
        return self.xarray_data.data

    @property
    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        dtype: np.dtype
            Data-type of the image array's elements.
        """
        return self.xarray_dask_data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        shape: Tuple[int, ...]
            Tuple of the image array's dimensions.
        """
        return self.xarray_dask_data.shape

    @property
    def dims(self) -> biob.dimensions.Dimensions:
        """
        Returns
        -------
        dims: dimensions.Dimensions
            Object with the paired dimension names and their sizes.
        """
        if self._dims is None:
            self._dims = biob.dimensions.Dimensions(
                dims=self.xarray_dask_data.dims, shape=self.shape
            )

        return self._dims

    def get_image_dask_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> da.Array:
        """
        Get specific dimension image data out of an image as a dask array.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: dimensions.DEFAULT_DIMENSION_ORDER (with or without Samples)

        kwargs: Any
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the dimension_order_out.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indices
              desired. D should be present in the dimension_order_out.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indices
              desired. D should be present in the dimension_order_out.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the dimension_order_out.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the dimension_order_out.

        Returns
        -------
        data: da.Array
            The image data with the specified dimension ordering.

        Examples
        --------
        Specific index selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... c1 = img.get_image_dask_data("ZYX", C=1)

        List of index selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_second = img.get_image_dask_data("CZYX", C=[0, 1])

        Tuple of index selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_last = img.get_image_dask_data("CZYX", C=(0, -1))

        Range of index selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_three = img.get_image_dask_data("CZYX", C=range(3))

        Slice selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... every_other = img.get_image_dask_data("CZYX", C=slice(0, -1, 2))

        Notes
        -----
        If a requested dimension is not present in the data the dimension is
        added with a depth of 1.

        See `bioio_base.transforms.reshape_data` for more details.
        """
        # If no out orientation, simply return current data as dask array
        if dimension_order_out is None and not kwargs:
            return self.dask_data

        # Transform and return
        return biob.transforms.reshape_data(
            data=self.dask_data,
            given_dims=self.dims.order,
            return_dims=dimension_order_out,
            **kwargs,
        )

    def get_image_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> np.ndarray:
        """
        Read the image as a numpy array then return specific dimension image data.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: dimensions.DEFAULT_DIMENSION_ORDER (with or without Samples)

        kwargs: Any
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the dimension_order_out.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indices
              desired. D should be present in the dimension_order_out.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indices
              desired. D should be present in the dimension_order_out.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the dimension_order_out.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the dimension_order_out.

        Returns
        -------
        data: np.ndarray
            The image data with the specified dimension ordering.

        Examples
        --------
        Specific index selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... c1 = img.get_image_data("ZYX", C=1)

        List of index selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_second = img.get_image_data("CZYX", C=[0, 1])

        Tuple of index selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_last = img.get_image_data("CZYX", C=(0, -1))

        Range of index selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_three = img.get_image_dask_data("CZYX", C=range(3))

        Slice selection

        >>> img = BioImage("s_1_t_1_c_10_z_20.ome.tiff")
        ... every_other = img.get_image_data("CZYX", C=slice(0, -1, 2))

        Notes
        -----
        * If a requested dimension is not present in the data the dimension is
          added with a depth of 1.
        * This will preload the entire image before returning the requested data.

        See `bioio_base.transforms.reshape_data` for more details.
        """
        # If no out orientation, simply return current data as dask array
        if dimension_order_out is None and not kwargs:
            return self.data

        # Transform and return
        return biob.transforms.reshape_data(
            data=self.data,
            given_dims=self.dims.order,
            return_dims=dimension_order_out,
            **kwargs,
        )

    def get_stack(self, **kwargs: Any) -> np.ndarray:
        """
        Get all scenes stacked in to a single array.

        Returns
        -------
        stack: np.ndarray
            The fully stacked array. This can be 6+ dimensions with Scene being
            the first dimension.
        kwargs: Any
            Extra keyword arguments that will be passed down to the
            generate stack function.

        See Also
        --------
        bioio_base.transforms.generate_stack:
            Underlying function for generating various scene stacks.
        """
        return biob.transforms.generate_stack(self, "data", **kwargs)

    def get_dask_stack(self, **kwargs: Any) -> da.Array:
        """
        Get all scenes stacked in to a single array.

        Returns
        -------
        stack: da.Array
            The fully stacked array. This can be 6+ dimensions with Scene being
            the first dimension.
        kwargs: Any
            Extra keyword arguments that will be passed down to the
            generate stack function.

        See Also
        --------
        bioio_base.transforms.generate_stack:
            Underlying function for generating various scene stacks.
        """
        return biob.transforms.generate_stack(self, "dask_data", **kwargs)

    def get_xarray_stack(self, **kwargs: Any) -> xr.DataArray:
        """
        Get all scenes stacked in to a single array.

        Returns
        -------
        stack: xr.DataArray
            The fully stacked array. This can be 6+ dimensions with Scene being
            the first dimension.
        kwargs: Any
            Extra keyword arguments that will be passed down to the
            generate stack function.

        See Also
        --------
        bioio_base.transforms.generate_stack:
            Underlying function for generating various scene stacks.

        Notes
        -----
        When requesting an xarray stack, the first scene's coordinate planes
        are used for the returned xarray DataArray object coordinate planes.
        """
        return biob.transforms.generate_stack(self, "xarray_data", **kwargs)

    def get_xarray_dask_stack(self, **kwargs: Any) -> xr.DataArray:
        """
        Get all scenes stacked in to a single array.

        Returns
        -------
        stack: xr.DataArray
            The fully stacked array. This can be 6+ dimensions with Scene being
            the first dimension.
        kwargs: Any
            Extra keyword arguments that will be passed down to the
            generate stack function.

        See Also
        --------
        bioio_base.transforms.generate_stack:
            Underlying function for generating various scene stacks.

        Notes
        -----
        When requesting an xarray stack, the first scene's coordinate planes
        are used for the returned xarray DataArray object coordinate planes.
        """
        return biob.transforms.generate_stack(self, "xarray_dask_data", **kwargs)

    @property
    def metadata(self) -> Any:
        """
        Returns
        -------
        metadata: Any
            Passthrough to the base image reader metadata property.
            For more information, see the specific image format reader you are using
            for details on its metadata property.
        """
        return self.reader.metadata

    @property
    def ome_metadata(self) -> OME:
        """
        Returns
        -------
        metadata: OME
            The original metadata transformed into the OME specfication.
            This likely isn't a complete transformation but is guarenteed to
            be a valid transformation.

        Raises
        ------
        NotImplementedError
            No metadata transformer available.
        """
        return self.reader.ome_metadata

    @property
    def channel_names(self) -> List[str]:
        """
        Returns
        -------
        channel_names: List[str]
            Using available metadata, the list of strings representing channel names.
        """
        # Unlike the base readers, the BioImage guarantees a Channel dim
        return list(
            self.xarray_dask_data[biob.dimensions.DimensionNames.Channel].values
        )

    @property
    def physical_pixel_sizes(self) -> biob.types.PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.

        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        return self.reader.physical_pixel_sizes

    @property
    def scale(self) -> biob.types.Scale:
        """
        Returns
        -------
        scale: Scale
            A Scale object constructed from the Reader's time_interval and
            physical_pixel_sizes.

        Notes
        -----
        * Combines temporal and spatial scaling information into a single object.
        * The channel scaling (`C`) is not derived from metadata and defaults to None.
        """
        return self.reader.scale

    @property
    def time_interval(self) -> biob.types.TimeInterval:
        """
        Returns
        -------
        sizes: Time Interval
            Using available metadata, this float represents the time interval for
            dimension T.

        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        return self.reader.time_interval

    @property
    def standard_metadata(self) -> biob.standard_metadata.StandardMetadata:
        """
        Return a set of standardized metadata. The possible
        fields are predefined by the StandardMetadata dataclass.
        """
        return self.reader.standard_metadata

    def get_mosaic_tile_position(
        self, mosaic_tile_index: int, **kwargs: int
    ) -> Tuple[int, int]:
        """
        Get the absolute position of the top left point for a single mosaic tile.

        Parameters
        ----------
        mosaic_tile_index: int
            The index for the mosaic tile to retrieve position information for.
        kwargs: int
            The keywords below allow you to specify the dimensions that you wish
            to match. If you under-specify the constraints you can easily
            end up with a massive image stack.
                       Z = 1   # The Z-dimension.
                       C = 2   # The C-dimension ("channel").
                       T = 3   # The T-dimension ("time").

        Returns
        -------
        top: int
            The Y coordinate for the tile position.
        left: int
            The X coordinate for the tile position.

        Raises
        ------
        UnexpectedShapeError
            The image has no mosaic dimension available.
        """
        return self.reader.get_mosaic_tile_position(mosaic_tile_index, **kwargs)

    def get_mosaic_tile_positions(self, **kwargs: int) -> List[Tuple[int, int]]:
        """
        Get the absolute positions of the top left points for each mosaic tile
        matching the specified dimensions and current scene.

        Parameters
        ----------
        kwargs: int
            The keywords below allow you to specify the dimensions that you wish
            to match. If you under-specify the constraints you can easily
            end up with a massive image stack.
                       Z = 1   # The Z-dimension.
                       C = 2   # The C-dimension ("channel").
                       T = 3   # The T-dimension ("time").
                       M = 4   # The mosaic tile index

        Returns
        -------
        mosaic_tile_positions: List[Tuple[int, int]]
            List of the Y and X coordinate for the tile positions.

        Raises
        ------
        UnexpectedShapeError
            The image has no mosaic dimension available.
        NotImplementedError
            Unable to combine M dimension with other dimensions when finding
            tiles matching kwargs
        """
        if biob.dimensions.DimensionNames.MosaicTile in kwargs:
            # Don't support getting positions by M + another dim
            if len(kwargs) != 1:
                other_keys = {
                    key
                    for key in kwargs
                    if key != biob.dimensions.DimensionNames.MosaicTile
                }
                raise NotImplementedError(
                    "Unable to determine appropriate position using mosaic tile "
                    + "index (M) combined with other dimensions "
                    + f"(including {other_keys})"
                )

            return [
                self.get_mosaic_tile_position(
                    kwargs[biob.dimensions.DimensionNames.MosaicTile]
                )
            ]

        return self.reader.get_mosaic_tile_positions(**kwargs)

    @property
    def mosaic_tile_dims(self) -> Optional[biob.dimensions.Dimensions]:
        """
        Returns
        -------
        tile_dims: Optional[Dimensions]
            The dimensions for each tile in the mosaic image.
            If the image is not a mosaic image, returns None.
        """
        return self.reader.mosaic_tile_dims

    def save(
        self,
        uri: biob.types.PathLike,
        select_scenes: Optional[Union[List[str], Tuple[str, ...]]] = None,
    ) -> None:
        """
        Saves the file data to OME-TIFF format with general naive best practices.

        Parameters
        ----------
        uri: biob.types.PathLike
            The URI or local path for where to save the data.
            Note: Can only write to local file systems.
        select_scenes: Optional[Union[List[str], Tuple[str, ...]]]
            Which scenes in the image to save to the file.
            Default: None (save all scenes)

        Notes
        -----
        See `bioio.writers.OmeTiffWriter` for more in-depth specification
        and the `bioio.writers` module as a whole for list of all available
        file writers.

        When reading in the produced OME-TIFF file, scenes IDs may have changed.
        This is due to how certain file and metadata formats do or do-not have IDs
        and simply names. In converting to OME-TIFF we will always store the scene
        ids in each Image's name attribute but IDs will be generated. The order of the
        scenes will be the same (or whatever order was specified / provided).
        """
        from .writers import OmeTiffWriter

        # Get all parameters as dict of lists, or static because of unchanging values
        datas: List[biob.types.ArrayLike] = []
        dim_orders: List[Optional[str]] = []
        channel_names: List[Optional[List[str]]] = []
        image_names: List[Optional[str]] = []
        physical_pixel_sizes: List[biob.types.PhysicalPixelSizes] = []

        # Get selected scenes / handle None scenes
        if select_scenes is None:
            select_scenes = self.scenes

        # Iter through scenes to get data
        for scene in select_scenes:
            self.set_scene(scene)

            # Append this scene details
            datas.append(self.dask_data)
            dim_orders.append(self.dims.order)
            channel_names.append(self.channel_names)
            image_names.append(self.current_scene)
            physical_pixel_sizes.append(self.physical_pixel_sizes)

        # Save all selected scenes
        OmeTiffWriter.save(
            data=datas,
            uri=uri,
            dim_order=dim_orders,
            channel_names=channel_names,
            image_name=image_names,
            physical_pixel_sizes=physical_pixel_sizes,
        )

    def __str__(self) -> str:
        if self._plugin is not None:
            return (
                f"<BioImage ["
                f"plugin: {self._plugin.entrypoint.name} installed "
                f"at {datetime.datetime.fromtimestamp(self._plugin.timestamp)}, "
                f"Image-is-in-Memory: {self._xarray_data is not None}"
                f"]>"
            )

        return f"<BioImage [Image-is-in-Memory: {self._xarray_data is not None}]>"

    def __repr__(self) -> str:
        return str(self)


def _construct_img(
    image: biob.types.ImageLike, scene_id: Optional[str] = None, **kwargs: Any
) -> BioImage:
    # Construct image
    img = BioImage(image, **kwargs)

    # Select scene
    if scene_id is not None:
        img.set_scene(scene_id)

    return img


def imread_xarray_dask(
    image: biob.types.ImageLike,
    scene_id: Optional[str] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Read image as a delayed xarray DataArray.

    Parameters
    ----------
    image: biob.types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    scene_id: Optional[str]
        An optional scene id to create the DataArray with.
        Default: None (First Scene)
    kwargs: Any
        Extra keyword arguments to be passed down to the BioImage and Reader subclass.

    Returns
    -------
    data: xr.DataArray
        The image read, scene selected, and returned as an BioIO standard shaped delayed
        xarray DataArray.
    """
    return _construct_img(image, scene_id, **kwargs).xarray_dask_data


def imread_dask(
    image: biob.types.ImageLike,
    scene_id: Optional[str] = None,
    **kwargs: Any,
) -> da.Array:
    """
    Read image as a delayed dask array.

    Parameters
    ----------
    image: biob.types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    scene_id: Optional[str]
        An optional scene id to create the dask array with.
        Default: None (First Scene)
    kwargs: Any
        Extra keyword arguments to be passed down to the BioImage and Reader subclass.

    Returns
    -------
    data: da.core.Array
        The image read, scene selected, and returned as an BioIO standard shaped delayed
        dask array.
    """

    return _construct_img(image, scene_id, **kwargs).dask_data


def imread_xarray(
    image: biob.types.ImageLike,
    scene_id: Optional[str] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Read image as an in-memory xarray DataArray.

    Parameters
    ----------
    image: biob.types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    scene_id: Optional[str]
        An optional scene id to create the DataArray with.
        Default: None (First Scene)
    kwargs: Any
        Extra keyword arguments to be passed down to the BioImage and Reader subclass.

    Returns
    -------
    data: xr.DataArray
        The image read, scene selected, and returned as an BioIO standard shaped
        in-memory DataArray.
    """
    return _construct_img(image, scene_id, **kwargs).xarray_data


def imread(
    image: biob.types.ImageLike,
    scene_id: Optional[str] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Read image as a numpy array.

    Parameters
    ----------
    image: biob.types.ImageLike
        A string, Path, fsspec supported URI, or arraylike to read.
    scene_id: Optional[str]
        An optional scene id to create the numpy array with.
        Default: None (First Scene)
    kwargs: Any
        Extra keyword arguments to be passed down to the BioImage and Reader subclass.

    Returns
    -------
    data: np.ndarray
        The image read, scene selected, and returned as an bioio standard shaped
        np.ndarray.
    """
    return _construct_img(image, scene_id, **kwargs).data
