# TODO: PLACEHOLDER

from typing import Any, Dict, List, Optional, Union

import bioio_base as bb
import ome_types


class OmeTiffWriter:
    @staticmethod
    def save(
        data: Union[List[bb.types.ArrayLike], bb.types.ArrayLike],
        uri: bb.types.PathLike,
        dim_order: Optional[Union[str, List[Union[str, None]]]] = None,
        ome_xml: Optional[Union[str, ome_types.model.OME]] = None,
        channel_names: Optional[Union[List[str], List[Optional[List[str]]]]] = None,
        image_name: Optional[Union[str, List[Union[str, None]]]] = None,
        physical_pixel_sizes: Optional[
            Union[bb.types.PhysicalPixelSizes, List[bb.types.PhysicalPixelSizes]]
        ] = None,
        channel_colors: Optional[
            Union[List[List[int]], List[Optional[List[List[int]]]]]
        ] = None,
        fs_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("TODO")
