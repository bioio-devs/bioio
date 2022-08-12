# -*- coding: utf-8 -*-

"""Top-level package for bioio."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioio")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"
