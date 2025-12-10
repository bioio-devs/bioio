#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    get_args,
)

import semver

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points, requires
else:
    from importlib.metadata import entry_points, EntryPoint, requires

from bioio_base.reader import Reader
from bioio_base.reader_metadata import ReaderMetadata
from bioio_base.types import ArrayLike, ImageLike, PathLike

from .array_like_reader import ArrayLikeReader, ArrayLikeReaderMetadata

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

BIOIO_DIST_NAME = "bioio"
BIOIO_BASE_DIST_NAME = "bioio-base"


class PluginEntry(NamedTuple):
    entrypoint: EntryPoint
    metadata: ReaderMetadata


# global cache of plugins
plugins_by_ext_cache: OrderedDict[str, List[PluginEntry]] = OrderedDict()


def _normalize_extensions(exts: Sequence[str]) -> List[str]:
    """
    Normalize and deduplicate a sequence of extensions.

    - Lowercases all entries
    - Ensures each extension starts with '.'
    - Removes duplicates while preserving first-seen order
    """
    normalized: List[str] = []
    seen = set()

    for ext in exts:
        e = ext.lower()
        if not e.startswith("."):
            e = "." + e

        if e not in seen:
            seen.add(e)
            normalized.append(e)

    return normalized


def _count_extension_families(exts: Sequence[str]) -> int:
    """
    Given a list of *normalized* extensions (lowercase, leading '.'), compute
    how many distinct 'families' of extensions this plugin effectively supports.

    Two extensions belong to the same family if one is a suffix of the other.

    Example
    -------
    [".ome.tiff", ".tiff", ".ome.tif", ".tif"] -> 2 families:
        {".ome.tiff", ".tiff"}, {".ome.tif", ".tif"}
    """
    families: List[List[str]] = []

    for ext in exts:
        matching_families: List[int] = []

        for i, fam in enumerate(families):
            # Any overlap by suffix membership → same family
            if any(ext.endswith(e) or e.endswith(ext) for e in fam):
                matching_families.append(i)

        if not matching_families:
            # New family
            families.append([ext])
        else:
            # Merge into the first matching family
            base_idx = matching_families[0]
            families[base_idx].append(ext)

            # If we matched multiple existing families, merge them
            for idx in reversed(matching_families[1:]):
                families[base_idx].extend(families[idx])
                del families[idx]

    return len(families)


def check_type(image: ImageLike, reader_class: Reader) -> bool:
    """
    Check if the provided image is compatible with the specified reader class.

    Parameters
    ----------
    image : ImageLike
        The image to be checked. It can be a PathLike, ArrayLike, MetaArrayLike,
        a list of MetaArrayLike, or a list of PathLike.

    reader_class : Reader
        The reader class to be checked against.

    Returns
    -------
    bool
        Returns True if the image is compatible with the reader class, False otherwise.
    """
    arraylike_types = tuple(get_args(ArrayLike))
    pathlike_types = tuple(get_args(PathLike))

    # Check if image is type ArrayLike or list of ArrayLike and reader
    # is not an ArrayLikeReader
    if (
        isinstance(image, arraylike_types)
        or (
            isinstance(image, list)
            and all(isinstance(item, arraylike_types) for item in image)
        )
    ) and not (reader_class is ArrayLikeReader):
        return False

    # Check if image is type PathLike or list of PathLike and reader
    # is an ArrayLikeReader
    if (
        isinstance(image, pathlike_types)
        or (
            isinstance(image, list)
            and all(isinstance(item, pathlike_types) for item in image)
        )
    ) and (reader_class is ArrayLikeReader):
        return False

    return True


def get_array_like_plugin() -> PluginEntry:
    """
    Create and return a PluginEntry for ArrayLikeReader.
    """
    entrypoint = EntryPoint(
        name="ArrayLikeReader",
        group="bioio.readers",
        value="bioio.array_like_reader:ArrayLikeReader",
    )
    metadata = ArrayLikeReaderMetadata()
    return PluginEntry(entrypoint=entrypoint, metadata=metadata)


def order_plugins_by_priority(
    plugins: List[PluginEntry],
    plugin_priority: Optional[Sequence[Type[Reader]]] = None,
) -> List[PluginEntry]:
    """
    Reorder a list of PluginEntry objects according to a user-provided
    list of Reader classes.

    Parameters
    ----------
    plugins : List[PluginEntry]
        The candidate plugins for a given extension.
    plugin_priority : Sequence[Type[Reader]], optional
        Reader classes that should be preferred first, e.g.

            from bioio_czi import Reader as CziReader
            from bioio_tifffile import Reader as TiffReader

            plugin_priority = [CziReader, TiffReader]

    Returns
    -------
    List[PluginEntry]
        Prioritized plugin list.
    """
    if not plugin_priority:
        return plugins

    priority_index = {cls: i for i, cls in enumerate(plugin_priority)}

    # From the docs:
    # "The built-in [sorted()](https://docs.python.org/3/library/functions.html#sorted)
    # function is guaranteed to be stable. A sort is stable if it guarantees not to
    # change the relative order of elements that compare equal."
    return sorted(
        plugins,
        key=lambda p: priority_index.get(p.metadata.get_reader(), len(priority_index)),
    )


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
    Gather a ordered mapping from supported file extensions to installed plugins.

    The return value is an OrderedDict:

        { normalized_extension (str) -> [PluginEntry, ...] }

    where both the **extension keys** and the **plugin lists** are ordered to
    encode BioIO's default selection policy.

    Ordering semantics
    ------------------
    1. Extension ordering (which extension group is checked first)
       ----------------------------------------------------------
       Extension keys are sorted by **descending length of the extension
       string**. This controls the order in which *extension groups* are
       considered when matching a path, independent of which plugins live
       under each key.

       Example:
         Keys: [".ome.tiff", ".tiff", ".tif"]
         Ordered as: [".ome.tiff", ".tiff", ".tif"]

       For a path ending with "file.ome.tiff", we will:
         * first consider plugins registered under the ".ome.tiff" key
         * then, if needed, fall back to the ".tiff" key
         * and finally ".tif", etc.

       This ordering is purely about the suffix string itself; it does **not**
       depend on how many extensions any plugin declares.

    2. Plugin ordering within each extension (which reader is tried first)
       -------------------------------------------------------------------
       For all plugins that advertise a given normalized extension key,
       we compute:

         * ``family_count`` — the number of **extension families** supported by
           that plugin, using ``_count_extension_families`` on its normalized
           extension list. Two extensions are in the same family if the
           dot-separated segment list of one is a suffix of the other's.

           Examples (normalized):
             [".ome.tiff", ".tiff"]  -> 1 family   ({".ome.tiff", ".tiff"})
             [".ome.tif", ".tif"]    -> 1 family   ({".ome.tif",  ".tif"})
             [".tif", ".tiff"]       -> 2 families ({".tif"}, {".tiff"})
             [".a.b", ".b"]          -> 1 family   ({".a.b", ".b"})
             [".ab", ".a"]           -> 2 families ({".ab"},  {".a"})

         * ``raw_ext_count`` — the number of extensions returned directly by
           ``ReaderMetadata.get_supported_extensions()`` (before normalization).

       Plugins for a given extension key are then sorted by:
         1) increasing ``family_count``      (fewer families → more specific)
         2) increasing ``raw_ext_count``     (fewer declared extensions → more focused)
         3) alphabetical entrypoint name     (deterministic tie-breaker)

       This means that, **for the same extension key**, a plugin that handles
       fewer, tightly related families and declares fewer total extensions will
       be tried *before* a broader, more generic reader.

    Notes
    -----
    * Before including a plugin, we check its declared ``bioio-base`` version
      range against the version required by the core ``bioio`` distribution and
      skip plugins that are incompatible.
    * This default ordering controls BioIO's automatic choice of reader based
      on extension. A user can still override the order of candidate plugins
      for a given file via the ``reader`` argument on ``BioImage``.
    """
    if use_cache and plugins_by_ext_cache:
        return plugins_by_ext_cache.copy()

    eps = entry_points(group="bioio.readers")

    # Mapping of extension -> list[PluginEntry]
    plugins_by_ext: Dict[str, List[PluginEntry]] = {}

    # Per-plugin specificity metrics
    plugin_family_counts: Dict[PluginEntry, int] = {}
    plugin_raw_ext_counts: Dict[PluginEntry, int] = {}

    (
        min_compatible_bioio_base_version,
        _,
    ) = get_dependency_version_range_for_distribution(
        BIOIO_DIST_NAME, BIOIO_BASE_DIST_NAME
    )

    for ep in eps:
        (
            _,
            max_bioio_base_version_for_plugin,
        ) = get_dependency_version_range_for_distribution(ep.name, BIOIO_BASE_DIST_NAME)

        # Skip plugins whose maximum supported bioio-base version is older
        # than the minimum that this bioio distribution requires.
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
                f"Plugin `{ep.name}` does not meet "
                f"minimum `{BIOIO_BASE_DIST_NAME}` of "
                f"{min_compatible_bioio_base_version}, ignoring"
            )
            continue

        # ReaderMetadata knows how to instantiate the actual Reader
        reader_meta = ep.load().ReaderMetadata

        # Raw + normalized extensions
        raw_exts = reader_meta.get_supported_extensions()
        normalized_exts = _normalize_extensions(raw_exts)

        family_count = _count_extension_families(normalized_exts)
        raw_ext_count = len(raw_exts)

        plugin_entry = PluginEntry(ep, reader_meta)

        plugin_family_counts[plugin_entry] = family_count
        plugin_raw_ext_counts[plugin_entry] = raw_ext_count

        # Register this plugin for each normalized extension it claims
        for ext in normalized_exts:
            plugins_by_ext.setdefault(ext, []).append(plugin_entry)

    # Order plugins within each extension:
    #   1) fewer extension families (more specific)
    #   2) fewer raw declared extensions
    #   3) alphabetical entrypoint name
    for ext, plugin_list in plugins_by_ext.items():
        plugin_list.sort(
            key=lambda p: (
                plugin_family_counts[p],
                plugin_raw_ext_counts[p],
                p.entrypoint.name.lower(),
            )
        )

    # Order the extension keys by length (descending) so more specific suffixes
    # (e.g., ".ome.tiff") are checked before shorter ones (e.g., ".tiff").
    plugins_by_ext_ordered: OrderedDict[str, List[PluginEntry]] = OrderedDict(
        sorted(
            plugins_by_ext.items(),
            key=lambda ext_and_plugins: len(ext_and_plugins[0]),
            reverse=True,
        )
    )

    # Save copy of plugins to cache then return
    plugins_by_ext_cache.clear()
    plugins_by_ext_cache.update(plugins_by_ext_ordered)

    return plugins_by_ext_ordered


def dump_plugins(use_cache: bool = True) -> None:
    """
    Report information about plugins currently installed

    Parameters
    ----------
    use_cache : bool
        Whether to use the cached plugins list. Mainly exposed for testing purposes.
    """
    plugins_by_ext = get_plugins(use_cache=use_cache)
    plugin_set = set()
    for _, plugins in plugins_by_ext.items():
        plugin_set.update(plugins)

    for plugin in plugin_set:
        ep = plugin.entrypoint
        print(ep.name)

        # Unpack dist
        dist = ep.dist
        if dist is not None:
            author = dist.metadata.json.get("author")
            license = dist.metadata.json.get("license")

            print(f'  Author  : {author if author is not None else "Not Specified"}')
            print(f"  Version : {dist.version}")
            print(f'  License : {license if license is not None else "Not Specified"}')

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

        reader_meta = plugin.metadata
        exts = ", ".join(reader_meta.get_supported_extensions())
        print(f"  Supported Extensions : {exts}")
    print("Plugins for extensions:")
    sorted_exts = sorted(plugins_by_ext.keys())
    for ext in sorted_exts:
        plugins = plugins_by_ext[ext]
        print(f"{ext}: {plugins}")


@dataclass
class PluginSupport:
    """Dataclass for reporting a plugins support for a specific image."""

    supported: bool
    error: Optional[str]


def _check_plugin_support(
    plugin: PluginEntry, image: ImageLike, fs_kwargs: Dict[str, Any] = {}
) -> PluginSupport:
    """Helper function to check if a plugin supports the given image."""
    try:
        ReaderClass = plugin.metadata.get_reader()
        supported = ReaderClass.is_supported_image(image=image, fs_kwargs=fs_kwargs)
        return PluginSupport(supported=supported, error=None)
    except Exception as e:
        return PluginSupport(supported=False, error=str(e))


def plugin_feasibility_report(
    image: ImageLike,
    fs_kwargs: Dict[str, Any] = {},
    use_plugin_cache: bool = False,
    **kwargs: Any,
) -> Dict[str, PluginSupport]:
    """
    Generate a feasibility report for each plugin,
    determining if it can handle the specified image.

    For debugging purposes, this function checks every installed plugin’s
    `is_supported_image(...)` implementation — even if the plugin would
    *not* normally be selected by BioImage’s extension-based routing.

    A warning is logged if a plugin *can* read the file but the file’s
    extension is NOT listed in that plugin’s advertised supported extensions.
    In such cases, BioImage will NOT auto-select that plugin for the file,
    but the user may explicitly choose it via the `reader=` parameter.
    """
    plugins_by_ext = get_plugins(use_cache=use_plugin_cache)
    feasibility_report: Dict[str, PluginSupport] = {}

    ext = None
    if isinstance(image, (str, Path)):
        # Strip any query parameters, then extract suffix
        clean_path = str(image).split("?")[0]
        ext = Path(clean_path).suffix.lower()

    # Check each plugin for support
    for plugins in plugins_by_ext.values():
        for plugin in plugins:
            plugin_name = plugin.entrypoint.name
            support = _check_plugin_support(plugin, image, fs_kwargs)
            feasibility_report[plugin_name] = support

            if support.supported and ext is not None:
                advertised_exts = _normalize_extensions(
                    plugin.metadata.get_supported_extensions()
                )

                if ext not in advertised_exts:
                    ReaderClass = plugin.metadata.get_reader()

                    log.warning(
                        f"Plugin '{plugin_name}' CAN read the file '{image}', "
                        f"but the file extension '{ext}' is NOT listed in its "
                        f"get_supported_extensions(): {advertised_exts}.  BioImage "
                        f"will NOT auto-select this reader based on extension.\n"
                        f"To use this reader manually, instantiate BioImage with:\n"
                        f"    BioImage('{image}', reader={ReaderClass.__name__})"
                    )

    # Additional check for ArrayLike support
    try:
        supported = ArrayLikeReader.is_supported_image(image=image, fs_kwargs=fs_kwargs)
        feasibility_report["ArrayLike"] = PluginSupport(supported=supported, error=None)
    except Exception as e:
        feasibility_report["ArrayLike"] = PluginSupport(supported=False, error=str(e))

    # Log feasibility report in a readable format
    log.info("Feasibility Report Summary:")
    for name, status in feasibility_report.items():
        if status.error is not None:
            log.info(f"{name}: Unsupported - Error: {status.error}")
        else:
            log.info(f"{name}: {'Supported' if status.supported else 'Unsupported'}")

    return feasibility_report
