#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import typing

import numpy as np
from ome_types.model.simple_types import PixelType

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def generate_ome_channel_id(image_id: str, channel_id: typing.Union[str, int]) -> str:
    """
    Naively generates the standard OME channel ID using a provided ID.

    Parameters
    ----------
    image_id: str
        An image id to pull the image specific index from.
        See: `generate_ome_image_id` for more details.
    channel_id: Union[str, int]
        A string or int representing the ID for a channel.
        In the context of the usage of this function, this is usually used with the
        index of the channel.

    Returns
    -------
    ome_channel_id: str
        The OME standard for channel IDs.

    Notes
    -----
    ImageIds are usually: "Image:0", "Image:1", or "Image:N",
    ChannelIds are usually the combination of image index + channel index --
    "Channel:0:0" for the first channel of the first image for example.
    """
    # Remove the prefix 'Image:' to get just the index
    image_index = image_id.replace("Image:", "")
    return f"Channel:{image_index}:{channel_id}"


def generate_ome_image_id(image_id: typing.Union[str, int]) -> str:
    """
    Naively generates the standard OME image ID using a provided ID.

    Parameters
    ----------
    image_id: Union[str, int]
        A string or int representing the ID for an image.
        In the context of the usage of this function, this is usually used with the
        index of the scene / image.

    Returns
    -------
    ome_image_id: str
        The OME standard for image IDs.
    """
    return f"Image:{image_id}"


def dtype_to_ome_type(npdtype: np.dtype) -> PixelType:
    """
    Convert numpy dtype to OME PixelType

    Parameters
    ----------
    npdtype: numpy.dtype
        A numpy datatype.

    Returns
    -------
    ome_type: PixelType
        One of the supported OME Pixels types

    Raises
    ------
    ValueError
        No matching pixel type for provided numpy type.
    """
    ometypedict = {
        np.dtype(np.int8): PixelType.INT8,
        np.dtype(np.int16): PixelType.INT16,
        np.dtype(np.int32): PixelType.INT32,
        np.dtype(np.uint8): PixelType.UINT8,
        np.dtype(np.uint16): PixelType.UINT16,
        np.dtype(np.uint32): PixelType.UINT32,
        np.dtype(np.float32): PixelType.FLOAT,
        np.dtype(np.float64): PixelType.DOUBLE,
        np.dtype(np.complex64): PixelType.COMPLEXFLOAT,
        np.dtype(np.complex128): PixelType.COMPLEXDOUBLE,
    }
    ptype = ometypedict.get(npdtype)
    if ptype is None:
        raise ValueError(f"Ome utils can't resolve pixel type: {npdtype.name}")
    return ptype


def ome_to_numpy_dtype(ome_type: PixelType) -> np.dtype:
    """
    Convert OME PixelType to numpy dtype

    Parameters
    ----------
    ome_type: PixelType
        One of the supported OME Pixels types

    Returns
    -------
    npdtype: numpy.dtype
        A numpy datatype.

    Raises
    ------
    ValueError
        No matching numpy type for the provided pixel type.
    """
    ometypedict: typing.Dict[PixelType, np.dtype] = {
        PixelType.INT8: np.dtype(np.int8),
        PixelType.INT16: np.dtype(np.int16),
        PixelType.INT32: np.dtype(np.int32),
        PixelType.UINT8: np.dtype(np.uint8),
        PixelType.UINT16: np.dtype(np.uint16),
        PixelType.UINT32: np.dtype(np.uint32),
        PixelType.FLOAT: np.dtype(np.float32),
        PixelType.DOUBLE: np.dtype(np.float64),
        PixelType.COMPLEXFLOAT: np.dtype(np.complex64),
        PixelType.COMPLEXDOUBLE: np.dtype(np.complex128),
    }
    nptype = ometypedict.get(ome_type)
    if nptype is None:
        raise ValueError(f"Ome utils can't resolve pixel type: {ome_type.value}")
    return nptype
