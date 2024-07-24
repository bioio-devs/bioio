#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Tuple

import xarray as xr
from bioio_base.dimensions import Dimensions
from bioio_base.reader import Reader as BaseReader
from fsspec.spec import AbstractFileSystem

###############################################################################


class Reader(BaseReader):
    """
    The main class of each reader plugin. This class is subclass
    of the abstract class reader (BaseReader) in bioio-base.

    Parameters
    ----------
    image: Any
        Some type of object to read and follow the Reader specification.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}

    Notes
    -----
    It is up to the implementer of the Reader to decide which types they would like to
    accept (certain readers may not support buffers for example).

    """

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_dask_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[Dimensions] = None
    _metadata: Optional[Any] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    # Do not provide default value because
    # they may not need to be used by your reader (i.e. input param is an array)
    _fs: "AbstractFileSystem"
    _path: str

    # Required Methods

    def __init__(image: Any, **kwargs: Any):
        raise NotImplementedError()

    @staticmethod
    def _is_supported_image(fs: "AbstractFileSystem", path: str, **kwargs: Any) -> bool:
        raise NotImplementedError()

    @property
    def scenes(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    def _read_delayed(self) -> "xr.DataArray":
        raise NotImplementedError()

    def _read_immediate(self) -> "xr.DataArray":
        raise NotImplementedError()
