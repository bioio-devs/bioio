#!/usr/bin/env python

import sys
from typing import List

# Choose the right entry_points based on Python version
if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points as _entry_points
else:
    from importlib_metadata import entry_points as _entry_points

# Public API list
__all__: List[str] = []

# Discover all registered writers
_eps = _entry_points(group="bioio.writers")

for ep in _eps:
    cls = ep.load()
    globals()[ep.name] = cls
    __all__.append(ep.name)
