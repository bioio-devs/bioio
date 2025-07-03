#!/usr/bin/env python

"""Top-level package for bioio."""

from importlib.metadata import PackageNotFoundError, version

try:
    from bioio_base.dimensions import DimensionNames, Dimensions
    from bioio_base.standard_metadata import StandardMetadata
    from bioio_base.types import (
        ArrayLike,
        ImageLike,
        MetaArrayLike,
        PathLike,
        PhysicalPixelSizes,
        Scale,
        TimeInterval,
    )
    from bioio_base.writer import Writer
except ImportError:
    # Safe fallback during install/docs build
    _fallback_symbols = [
        "ArrayLike",
        "DimensionNames",
        "Dimensions",
        "ImageLike",
        "MetaArrayLike",
        "PathLike",
        "PhysicalPixelSizes",
        "Scale",
        "StandardMetadata",
        "TimeInterval",
        "Writer",
    ]
    globals().update({name: None for name in _fallback_symbols})


from .bio_image import BioImage
from .plugins import plugin_feasibility_report

try:
    __version__ = version("bioio")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown, Dan Toloudis, BioIO Contributors"
__email__ = "evamaxfieldbrown@gmail.com, danielt@alleninstitute.org"

__all__ = [
    "ArrayLike",
    "BioImage",
    "DimensionNames",
    "Dimensions",
    "ImageLike",
    "MetaArrayLike",
    "PathLike",
    "PhysicalPixelSizes",
    "Scale",
    "StandardMetadata",
    "TimeInterval",
    "Writer",
    "plugin_feasibility_report",
]
