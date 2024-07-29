# -*- coding: utf-8 -*-

"""Top-level package for bioio."""

from importlib.metadata import PackageNotFoundError, version

from .bio_image import BioImage
from .plugins import plugin_feasibility_report

try:
    __version__ = version("bioio")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown, Dan Toloudis, BioIO Contributors"
__email__ = "evamaxfieldbrown@gmail.com, danielt@alleninstitute.org"

__all__ = [
    "BioImage",
    "plugin_feasibility_report",
]
