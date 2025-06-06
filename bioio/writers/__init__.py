#!/usr/bin/env python

from importlib.metadata import entry_points as _entry_points
from typing import List

# Public API list
__all__: List[str] = []

# Discover all registered writers
_eps = _entry_points(group="bioio.writers")

for ep in _eps:
    cls = ep.load()
    globals()[ep.name] = cls
    __all__.append(ep.name)
