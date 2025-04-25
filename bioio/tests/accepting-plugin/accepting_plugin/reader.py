from typing import Any, Optional, Tuple

import xarray as xr
from bioio_base.dimensions import Dimensions
from bioio_base.reader import Reader as BaseReader
from fsspec.spec import AbstractFileSystem


class Reader(BaseReader):
    """
    Dummy reader plugin that pretends to read anything
    """

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_dask_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[Dimensions] = None
    _metadata: Optional[Any] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    _fs: "AbstractFileSystem"
    _path: str

    def __init__(*args: Any, **kwargs: Any):
        pass

    @staticmethod
    def _is_supported_image(fs: "AbstractFileSystem", path: str, **kwargs: Any) -> bool:
        return True

    @property
    def scenes(self) -> Tuple[str, ...]:
        return ()

    def _read_delayed(self) -> "xr.DataArray":
        return xr.DataArray([])

    def _read_immediate(self) -> "xr.DataArray":
        return xr.DataArray([])
