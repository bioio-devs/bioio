#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import semver

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points, requires
else:
    from importlib.metadata import entry_points, EntryPoint, requires

from typing import Dict, List, NamedTuple, Optional, Tuple

from bioio_base.reader_metadata import ReaderMetadata

###############################################################################

BIOIO_DIST_NAME = "bioio"
BIOIO_BASE_DIST_NAME = "bioio-base"


class PluginEntry(NamedTuple):
    entrypoint: EntryPoint
    metadata: ReaderMetadata
    timestamp: float


# global cache of plugins
plugins_by_ext_cache: Dict[str, List[PluginEntry]] = {}


def insert_sorted_by_timestamp(list: List[PluginEntry], item: PluginEntry) -> None:
    """
    Insert into list of PluginEntrys sorted by their timestamps (install dates)
    """
    for i, other in enumerate(list):
        if item.timestamp > other.timestamp:
            list.insert(i, item)
            return
    list.append(item)


def get_dependency_version_range_for_distribution(
    distribution_name: str, dependency_name: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieves the minimum and maximum versions of the dependency as specified
    by the given distribution.
    `None` as the minimum or maximum represents no minimum or maximum (respectively)
    """
    all_distribution_deps = requires(distribution_name) or []
    matching_distribution_deps = [
        dep for dep in all_distribution_deps if dep.startswith(dependency_name)
    ]
    if len(matching_distribution_deps) != 1:
        raise ValueError(
            f"Expected to find 1 `{dependency_name}` dependency for "
            f"`{distribution_name}`, instead found {len(matching_distribution_deps)}"
        )

    # Versions from importlib are formatted like "<dependency name> (<0.19,>=0.18)"
    # so this regex needs to find the inner part of the parentheses to get the
    # version specifications
    version_specification_match = re.findall(r"\(.*?\)", matching_distribution_deps[0])

    minimum_dependency_version = None
    maximum_dependency_version = None
    if version_specification_match:
        # A distribution can specify a minimum and maximum (or force an exact) version
        # so multiple versions need to be parsed here
        version_specifications = version_specification_match[0].split(",")
        for version_specification in version_specifications:
            version = "".join(
                [
                    char
                    for char in version_specification
                    if char.isnumeric() or char == "."
                ]
            )

            # Unfortunately not all versions are specified in full SemVer format
            # and instead can be like "0.19" and the "semver" package does not
            # handle these well so we can just fill in the missing .0
            # to make 0.19 -> 0.19.0
            while version.count(".") < 2:
                version += ".0"

            if "<" in version_specification or "==" in version_specification:
                maximum_dependency_version = version

            if ">" in version_specification or "==" in version_specification:
                minimum_dependency_version = version

    return minimum_dependency_version, maximum_dependency_version


def get_plugins(use_cache: bool) -> Dict[str, List[PluginEntry]]:
    """
    Gather a mapping from support file extensions
    to plugins installed that support said extension
    """
    if use_cache and plugins_by_ext_cache:
        return plugins_by_ext_cache.copy()

    plugins = entry_points(group="bioio.readers")

    # Mapping of extensions -> applicable plugins
    # note there can be multiple readers for the same extension
    plugins_by_ext: Dict[str, List[PluginEntry]] = {}

    (
        min_compatible_bioio_base_version,
        _,
    ) = get_dependency_version_range_for_distribution(
        BIOIO_DIST_NAME, BIOIO_BASE_DIST_NAME
    )
    for plugin in plugins:
        (
            _,
            max_bioio_base_version_for_plugin,
        ) = get_dependency_version_range_for_distribution(
            plugin.name, BIOIO_BASE_DIST_NAME
        )
        if (
            min_compatible_bioio_base_version is not None
            and max_bioio_base_version_for_plugin is not None
            and semver.compare(
                max_bioio_base_version_for_plugin,
                min_compatible_bioio_base_version,
            )
            < 0
        ):
            print(
                f"Plugin `{plugin.name}` does not meet "
                f"minimum `{BIOIO_BASE_DIST_NAME}` of "
                f"{min_compatible_bioio_base_version}, ignoring"
            )
        else:
            # ReaderMetadata knows how to instantiate the actual Reader
            reader_meta = plugin.load().ReaderMetadata
            if plugin.dist is not None:
                files = plugin.dist.files
                if files is not None:
                    firstfile = files[0]
                    timestamp = os.path.getmtime(Path(firstfile.locate()).parent)
                else:
                    print(f"No files found for plugin: '{plugin}'")
            else:
                print(f"Could not find distribution for plugin: '{plugin}'")
                timestamp = 0.0

            # Add plugin entry
            plugin_entry = PluginEntry(plugin, reader_meta, timestamp)
            for ext in plugin_entry.metadata.get_supported_extensions():
                if ext not in plugins_by_ext:
                    plugins_by_ext[ext] = [plugin_entry]
                    continue

                # insert in sorted order (sorted by most recently installed)
                pluginlist = plugins_by_ext[ext]
                insert_sorted_by_timestamp(pluginlist, plugin_entry)

    # Save copy of plugins to cache then return
    plugins_by_ext_cache.clear()
    plugins_by_ext_cache.update(plugins_by_ext)
    return plugins_by_ext


def dump_plugins() -> None:
    """
    Report information about plugins currently installed
    """
    plugins_by_ext = get_plugins(use_cache=True)
    plugin_set = set()
    for _, plugins in plugins_by_ext.items():
        plugin_set.update(plugins)

    for plugin in plugin_set:
        ep = plugin.entrypoint
        print(ep.name)

        # Unpack dist
        dist = ep.dist
        if dist is not None:
            print(f"  Author  : {dist.metadata['author']}")
            print(f"  Version : {dist.version}")
            print(f"  License : {dist.metadata['license']}")

            # Unpack files
            files = dist.files
            if files is not None:
                firstfile = files[0]
                t = datetime.fromtimestamp(
                    os.path.getmtime(Path(firstfile.locate()).parent)
                )
                print(f"  Date    : {t}")
        else:
            print("  No Distribution Found...")

            # print(f"  Description : {ep.dist.metadata['description']}")
        reader_meta = plugin.metadata
        exts = ", ".join(reader_meta.get_supported_extensions())
        print(f"  Supported Extensions : {exts}")
    print("Plugins for extensions:")
    sorted_exts = sorted(plugins_by_ext.keys())
    for ext in sorted_exts:
        plugins = plugins_by_ext[ext]
        print(f"{ext}: {plugins}")
