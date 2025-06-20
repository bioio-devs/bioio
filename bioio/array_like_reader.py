#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import constants, exceptions
from bioio_base.dimensions import (
    DEFAULT_DIMENSION_ORDER,
    DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
    DimensionNames,
)
from bioio_base.reader import Reader
from bioio_base.reader_metadata import ReaderMetadata
from bioio_base.types import MetaArrayLike, PhysicalPixelSizes

from .ome_utils import generate_ome_channel_id, generate_ome_image_id

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ArrayLikeReader(Reader):
    """
    A catch all for numpy, dask, or xarray to Reader interface.

    See Notes for more details.

    Parameters
    ----------
    image: Union[List[MetaArrayLike], MetaArrayLike]
        A single, numpy ndarray, dask Array, or xarray DataArray, or list of many.
        If provided a list, each item in the list will be exposed through the scene API.
        If provided an xarray DataArray alone or as an element of the list, the
        dim_order and channel_names kwargs are ignored if there non-xarray
        default dimension (or channel coordinate) information attached the xarray
        object. The channel and dimension information are updated independent of the
        other. If either is using xarray default values, they will be replaced by
        AICSImageIO defaults will be added.

    dim_order: Optional[Union[List[str], str]]
        A string of dimensions to be applied to all array(s) or a
        list of string dimension names to be mapped onto the list of arrays
        provided to image. I.E. "TYX".
        Default: None (guess dimensions for single array or multiple arrays)

    channel_names: Optional[Union[List[str], List[List[str]]]]
        A list of string channel names to be applied to all array(s) or a
        list of lists of string channel names to be mapped onto the list of arrays
        provided to image.
        Default: None (create OME channel IDs for names for single or multiple arrays)

    physical_pixel_sizes: Optional[
        Union[List[float], Dict[str, float], PhysicalPixelSizes]
    ]
        A specification of this image's physical pixel sizes. Can be provided as
        a list, dict or PhysicalPixelSizes object. If a list is passed, the assumed
        order is [Z, Y, X]. If a dict is passed, it must contain "Z", "Y" and "X"
        as keys.
        Default: None

    Raises
    ------
    exceptions.ConflictingArgumentsError
        Raised when the number of scenes provided is different from the number of items
        provided to the metadata parameters if as a list.

    exceptions.ConflictingArgumentsError
        Raised when channel_names is provided but the channel dimension was
        either not guessed or not provided in dim_order.

    ValueError
        Provided dim_order string or channel_names are not the same length as
        the number of dimensions or the size of the channel dimensions for the array at
        the matching index.

    Notes
    -----
    If you want to combine multiple numpy ndarrays or dask arrays with xarray DataArrays
    and attach metadata to all non-xarray.DataArrays, you can do so, the metadata for
    the xarray DataArray scenes will simply be ignored if it the non-xarray defaults.

    In such cases, it is recommended that you provide metadata for those scenes as
    None. I.E.

    >>> some_metadata_attached_xr = ...
    ... some_np = ...
    ... some_dask = ...
    ... reader = ArrayLikeReader(
    ...     image=[some_metadata_attached_xr, some_np, some_dask],
    ...     dim_order=[None, "CTYX", "ZTYX"],
    ...     channel_names=[None, ["A", "B", C"], None],
    ... )

    Will create a three scene ArrayLikeReader with the metadata and coordinate
    information filled xarray DataArray as the first scene, a numpy array with CTYX as
    the dimensions and ["A", "B", "C"] as the channel names, and a dask array with ZTYX
    as the dimensions and no channel names (as there is no channel dimension).
    """

    _scenes: Optional[Tuple[str, ...]] = None

    @staticmethod
    def _is_supported_image(  # type: ignore
        image: Union[List[MetaArrayLike], MetaArrayLike], *args: Any, **kwargs: Any
    ) -> bool:
        if isinstance(image, list):
            values = image
        else:
            values = [image]
        for scene in values:
            if isinstance(scene, (np.ndarray, da.Array, xr.DataArray)):
                continue
            else:
                raise exceptions.UnsupportedFileFormatError(
                    "ArrayLikeReader",
                    f"Unsupported image type: {type(image)}. "
                    + "ArrayLikeReader supported types are numpy ndarray, dask Array,"
                    + "or xarray DataArray.",
                )
        # If we got here, we have a supported image type
        return True

    @staticmethod
    def _guess_dim_order(shape: Tuple[int, ...]) -> str:
        """
        Given an image shape attempts to guess the dimension order.

        Parameters
        ----------
        shape: Tuple[int, ...]
            Tuple of the image array's dimensions.

        Returns
        -------
        dim_order: str
            The guessed dimension order.

        Raises
        ------
        exceptions.InvalidDimensionOrderingError
            Raised when the shape has more than dimensions than the ArrayLikeReader
            has a default order available to support.
        """
        if len(shape) > len(DEFAULT_DIMENSION_ORDER):
            if len(shape) > len(DEFAULT_DIMENSION_ORDER_WITH_SAMPLES):
                raise exceptions.InvalidDimensionOrderingError(
                    "Unable to guess dimension order for array-like images with "
                    + f"more than {len(DEFAULT_DIMENSION_ORDER_WITH_SAMPLES)} "
                    + f"dimensions. Found {len(shape)} dimensions."
                )

            return DEFAULT_DIMENSION_ORDER_WITH_SAMPLES[
                len(DEFAULT_DIMENSION_ORDER_WITH_SAMPLES) - len(shape) :
            ]

        return DEFAULT_DIMENSION_ORDER[len(DEFAULT_DIMENSION_ORDER) - len(shape) :]

    def __init__(
        self,
        image: Union[List[MetaArrayLike], MetaArrayLike],
        dim_order: Optional[Union[List[str], str]] = None,
        channel_names: Optional[Union[List[str], List[List[str]]]] = None,
        physical_pixel_sizes: Optional[
            Union[List[float], Dict[str, float], PhysicalPixelSizes]
        ] = None,
        **kwargs: Any,
    ):
        # Enforce valid image
        self._is_supported_image(image)

        # General note
        # Any time we do a `channel_names[0]` it's because the type check for
        # channel names is a List[List[str]], so by checking the first element we should
        # be getting back a list or a string. Anything else will error.

        # The result of all of this validation and pre-compute work is that at the end
        # of this init, we should have n-number of constructed xarray objects created
        # from the parameters provided that we then just pass-through to for all other
        # standard Reader operations

        # If metadata is attached as lists
        # Enforcing matching shape
        if isinstance(image, list):
            if isinstance(dim_order, list):
                # Check dim order
                if len(dim_order) != len(image):
                    raise exceptions.ConflictingArgumentsError(
                        f"ArrayLikeReader received a list of arrays to use as scenes "
                        f"but the provided list of dimension order strings is of "
                        f"different length. "
                        f"Number of provided scenes: {len(image)}, "
                        f"Number of provided dimension order strings: "
                        f"{len(dim_order)}"
                    )

            # Check channel names
            if channel_names is not None:
                if isinstance(channel_names[0], list):
                    if len(channel_names) != len(image):
                        raise exceptions.ConflictingArgumentsError(
                            f"ArrayLikeReader received a list of arrays to use as "
                            f"scenes but the provided list of channel names is "
                            f"of different length. "
                            f"Number of provided scenes: {len(image)}, "
                            f"Number of provided channel names: "
                            f"{len(channel_names)}"
                        )

        # If metadata is attached as singles
        # but many scenes provided, expand
        if isinstance(image, list):
            if dim_order is None or isinstance(dim_order, str):
                dim_order = [dim_order for i in range(len(image))]  # type: ignore
            if channel_names is None or isinstance(channel_names[0], str):
                channel_names = [  # type: ignore
                    channel_names for i in range(len(image))
                ]

        # Set all kwargs to lists for standard interface
        if not isinstance(image, list):
            image = [image]
        if not isinstance(dim_order, list):
            dim_order = [dim_order]  # type: ignore
        if channel_names is None:
            channel_names = [channel_names]  # type: ignore
        # Also wrap the channel names list if they were provided
        # but only a single scene was
        elif len(image) == 1 and not isinstance(channel_names[0], list):
            channel_names = [channel_names]  # type: ignore

        # Store image(s)
        self._all_scenes = image

        # Validate and store dims
        self._scene_dims_list = []
        for i, dims_string in enumerate(dim_order):
            this_scene = self._all_scenes[i]
            # Provided None, guess
            if dims_string is None:
                # Get dims from xarray
                if isinstance(this_scene, xr.DataArray):
                    # Guess for xarray when they guessed
                    # http://xarray.pydata.org/en/stable/data-structures.html?highlight=dim_n#creating-a-dataarray
                    # See "dim_N"
                    if this_scene.dims[0] == "dim_0":
                        log.debug(
                            "Overrode the default xarray dimensions for AICSImage "
                            "guessed dimensions."
                        )
                        # Rename the dimensions from "dim_N" to just the guess dim
                        # Update scene list in place
                        self._all_scenes[i] = this_scene.rename(
                            {
                                f"dim_{d_index}": d
                                for d_index, d in enumerate(
                                    self._guess_dim_order(this_scene.shape)
                                )
                            }
                        )

                # Guess for non xarray data
                self._scene_dims_list.append(self._guess_dim_order(this_scene.shape))

            # Provided Some, validate
            else:
                if len(dims_string) == len(this_scene.shape):
                    # Update dims for xarray
                    if isinstance(this_scene, xr.DataArray):
                        # Guess for xarray when they guessed
                        # http://xarray.pydata.org/en/stable/data-structures.html?highlight=dim_n#creating-a-dataarray
                        # See "dim_N"
                        if this_scene.dims[0] == "dim_0":
                            log.debug(
                                "Overrode the default xarray dimensions for AICSImage "
                                "provided dimensions."
                            )
                            # Rename the dimensions from "dim_N" to just the guess dim
                            # Update scene list in place
                            self._all_scenes[i] = this_scene.rename(
                                {
                                    f"dim_{d_index}": d
                                    for d_index, d in enumerate(dims_string)
                                }
                            )

                    self._scene_dims_list.append(dims_string)
                else:
                    raise ValueError(
                        f"Provided dimension string does not have the same amount of "
                        f"dimensions as the matching provided array. "
                        f"Provided array shape: {this_scene.shape}, "
                        f"Provided dimension string: {dims_string}"
                    )

        # Validate and store channel_names
        self._scene_channel_names = []
        for s_index, this_scene_channel_names in enumerate(
            channel_names  # type: ignore
        ):
            this_scene = self._all_scenes[s_index]
            this_scene_dims = self._scene_dims_list[s_index]

            # Create channel names is needed
            if DimensionNames.Channel in this_scene_dims:
                channel_dim_index = this_scene_dims.index(DimensionNames.Channel)

                # Handle xarray missing channel names
                if isinstance(this_scene, xr.DataArray):
                    if DimensionNames.Channel not in this_scene.coords:
                        # Use provided
                        if this_scene_channel_names is not None:
                            this_scene.coords[
                                DimensionNames.Channel
                            ] = this_scene_channel_names

                        # Generate
                        else:
                            set_channel_names = []
                            for c_index in range(this_scene.shape[channel_dim_index]):
                                image_id = generate_ome_image_id(s_index)
                                set_channel_names.append(
                                    generate_ome_channel_id(
                                        image_id=image_id, channel_id=c_index
                                    )
                                )

                            this_scene.coords[
                                DimensionNames.Channel
                            ] = set_channel_names

                # Provided None, generate
                if this_scene_channel_names is None:
                    this_scene_channels = []
                    for c_index in range(this_scene.shape[channel_dim_index]):
                        image_id = generate_ome_image_id(s_index)
                        this_scene_channels.append(
                            generate_ome_channel_id(
                                image_id=image_id, channel_id=c_index
                            )
                        )

                    self._scene_channel_names.append(this_scene_channels)

                # Provided Some, validate
                else:
                    if (
                        len(this_scene_channel_names)
                        != this_scene.shape[channel_dim_index]
                    ):
                        raise ValueError(
                            f"Provided channel names list does not match the size of "
                            f"channel dimension for the provided array. "
                            f"Provided array shape: {this_scene.shape}, "
                            f"Channel dimension size: "
                            f"{this_scene.shape[channel_dim_index]}, "
                            f"Provided channel names: {this_scene_channel_names}"
                        )
                    else:
                        self._scene_channel_names.append(this_scene_channel_names)

            # Raise error when channel names were provided when they shouldn't have been
            else:
                if this_scene_channel_names is not None:
                    raise ValueError(
                        f"Received channel names for array without channel dimension. "
                        f"Provided array shape: {this_scene.shape}, "
                        f"Provided (or guessed) dimensions: {this_scene_dims}, "
                        f"Provided channel names: {this_scene_channel_names}"
                    )
                else:
                    self._scene_channel_names.append(this_scene_channel_names)

        # Construct full xarrays
        # All data arrays in this list are dask Arrays
        self._xr_darrays = []
        for scene_data, dims, this_scene_channel_names in zip(
            self._all_scenes, self._scene_dims_list, self._scene_channel_names
        ):
            # Handle simple case of provided a DataArray
            if isinstance(scene_data, xr.DataArray):
                # Set metadata to point at existing attrs
                scene_data.attrs[constants.METADATA_UNPROCESSED] = scene_data.attrs

                # If the data backing the xarray is dask
                # just append
                if isinstance(scene_data.data, da.Array):
                    self._xr_darrays.append(scene_data)

                # If the data backing the xarray is numpy
                # copy the array but use dask from numpy
                else:
                    self._xr_darrays.append(
                        scene_data.copy(data=da.from_array(scene_data.data))
                    )

            # Handle non-xarray cases
            else:
                dims_list = list(dims)
                coords = {}
                if DimensionNames.Channel in dims_list:
                    coords[DimensionNames.Channel] = this_scene_channel_names

                # Handle dask
                if isinstance(scene_data, np.ndarray):
                    scene_data = da.from_array(scene_data)

                # Append the dask backed array
                self._xr_darrays.append(
                    xr.DataArray(
                        data=scene_data,
                        dims=dims_list,
                        coords=coords,
                        attrs={constants.METADATA_UNPROCESSED: None},
                    )
                )

        if isinstance(physical_pixel_sizes, PhysicalPixelSizes):
            self._physical_pixel_sizes = physical_pixel_sizes
        elif isinstance(physical_pixel_sizes, (list, tuple)):
            self._physical_pixel_sizes = PhysicalPixelSizes(*physical_pixel_sizes)
        elif isinstance(physical_pixel_sizes, dict):
            self._physical_pixel_sizes = PhysicalPixelSizes(**physical_pixel_sizes)
        else:
            self._physical_pixel_sizes = None

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            self._scenes = tuple(
                generate_ome_image_id(i) for i in range(len(self._all_scenes))
            )

        return self._scenes

    def _read_delayed(self) -> xr.DataArray:
        return self._xr_darrays[self.current_scene_index]

    def _read_immediate(self) -> xr.DataArray:
        return self._xr_darrays[self.current_scene_index].copy(
            data=self._xr_darrays[self.current_scene_index].data.compute()
        )

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
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
        return self._physical_pixel_sizes


class ArrayLikeReaderMetadata(ReaderMetadata):
    @staticmethod
    def get_supported_extensions() -> List[str]:
        # Since ArrayLikeReader supports numpy, dask, and xarray objects it does
        # not have file extensions
        return []

    @staticmethod
    def get_reader() -> Reader:
        return ArrayLikeReader
