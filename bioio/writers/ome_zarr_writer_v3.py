import sys

if sys.version_info < (3, 11):

    class OMEZarrWriter:
        """
        Stub for Python <3.11. Instantiating this will always error.
        """

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "OMEZarrWriter requires Python 3.11+ and zarr>=3.0.6. "
                "On Python 3.10 you can only use the basic writer."
            )

else:
    import math
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

    import numpy as np
    import zarr
    from zarr.codecs import BloscCodec, BloscShuffle

    def downsample_data(
        data: np.ndarray, factors: Tuple[int, int, int, int, int]
    ) -> np.ndarray:
        """
        Downsample a 5D array (T, C, Z, Y, X) by integer factors for Z, Y, X.

        Parameters
        ----------
        data : np.ndarray
            The input 5D array (T, C, Z, Y, X).
        factors : Tuple[int, int, int, int, int]
            Downsampling factors (t_factor, c_factor ignored, zf, yf, xf).

        Returns
        -------
        np.ndarray
            Downsampled array.
        """
        # TODO: There is probably a much more efficent way to do this

        # Only spatial downsampling (time & channel stay intact)
        zf, yf, xf = factors[2], factors[3], factors[4]
        T, C, Z, Y, X = data.shape

        # Clamp to avoid factors > dimension
        zf, yf, xf = min(zf, Z), min(yf, Y), min(xf, X)

        # Compute how many full blocks fit
        nz, ny, nx = Z // zf, Y // yf, X // xf

        # Trim off remainders so reshape works
        d = data[..., : nz * zf, : ny * yf, : nx * xf]

        # Reshape into blocks and average
        # new shape: (T,C,nz,zf,ny,yf,nx,xf)
        d = d.reshape(T, C, nz, zf, ny, yf, nx, xf)
        out = d.mean(axis=(3, 5, 7))

        # Round back if integer
        if not np.issubdtype(data.dtype, np.floating):
            out = np.rint(out).astype(data.dtype)
        return out

    def default_axes(
        names: List[str], types: List[str], units: List[Optional[str]]
    ) -> List[dict]:
        """
        Build axes metadata list from provided names, types, and units.

        Parameters
        ----------
        names : List[str]
            Axis names.
        types : List[str]
            Axis types.
        units : List[Optional[str]]
            Axis units (or None).

        Returns
        -------
        List[dict]
            List of axis metadata dictionaries.
        """

        # TODO: should this be a constant or a util or something?
        axes = []
        for n, t, u in zip(names, types, units):
            entry = {"name": n, "type": t}
            if u:
                entry["unit"] = u
            axes.append(entry)
        return axes

    class OMEZarrWriter:
        """
        OMEZarrWriter is a fully compliant OME-Zarr v0.5.0 writer built
        on Zarr v3 stores.
        """

        def __init__(
            self,
            store: Union[str, zarr.storage.StoreLike],
            shape: Tuple[int, int, int, int, int],
            dtype: Union[np.dtype, str],
            axes_names: List[str] = ["t", "c", "z", "y", "x"],
            axes_types: List[str] = ["time", "channel", "space", "space", "space"],
            axes_units: List[Optional[str]] = [None, None, None, None, None],
            axes_scale: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
            scale_factors: Tuple[int, int, int, int, int] = (1, 1, 2, 2, 2),
            num_levels: Optional[int] = None,
            chunks: Union[str, Tuple[int, ...], List[Tuple[int, ...]]] = "auto",
            shards: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
            compressor: Optional[BloscCodec] = None,
            image_name: str = "Image",
            channel_names: Optional[List[str]] = None,
            channel_colors: Optional[List[str]] = None,
            creator_info: Optional[dict] = None,
        ):
            """
            Initialize writer and automatically build axes and channel metadata.

            Parameters
            ----------
            store : Union[str, zarr.storage.StoreLike]
                Path or Zarr Store for output.
            shape : Tuple[int, int, int, int, int]
                Base image shape (T, C, Z, Y, X).
            dtype : Union[np.dtype, str]
                NumPy dtype of the image data.
            axes_names : List[str]
                Axis names.
            axes_types : List[str]
                Axis types.
            axes_units : List[Optional[str]]
                Axis units.
            axes_scale : List[float]
                Physical scale per axis.
            scale_factors : Tuple[int, int, int, int, int]
                Downsampling factors per axis.
            num_levels : Optional[int]
                Maximum number of pyramid levels.
            chunks : Union[str, Tuple[int, ...], List[Tuple[int, ...]]]
                Chunk specification or "auto".
            shards : Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]
                Shard specification.
            compressor : Optional[BloscCodec]
                Compressor codec to use.
            image_name : str
                Name for multiscale metadata.
            channel_names : Optional[List[str]]
                Channel labels.
            channel_colors : Optional[List[str]]
                Channel hex colors.
            creator_info : Optional[dict]
                Creator metadata.
            """
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)
            self.axes_scale = axes_scale
            self.scale_factors = scale_factors
            # Multiscale shapes
            self.level_shapes = self._compute_levels(num_levels)
            # Chunks & shards
            self.chunks = self._prepare_parameter(
                chunks, self._suggest_chunks, "chunks"
            )
            self.shards = self._prepare_parameter(
                shards, lambda s: s, "shards", required=False
            )
            # Zarr store & arrays
            self.root = self._init_store(store)
            self.datasets = self._create_arrays(compressor)

            # Build metadata
            axes_meta = default_axes(axes_names, axes_types, axes_units)
            channel_meta = None
            if channel_names or channel_colors:
                C = shape[1]
                channel_meta = []
                default_colors = [
                    "FF0000",
                    "00FF00",
                    "0000FF",
                    "FFFF00",
                    "FF00FF",
                    "00FFFF",
                ]
                for i in range(C):
                    color = (
                        channel_colors[i]
                        if channel_colors and i < len(channel_colors)
                        else default_colors[i % len(default_colors)]
                    )
                    label = (
                        channel_names[i]
                        if channel_names and i < len(channel_names)
                        else f"Channel {i}"
                    )
                    info = (
                        np.iinfo(self.dtype)
                        if np.issubdtype(self.dtype, np.integer)
                        else None
                    )
                    win_min, win_max = (
                        (int(info.min), int(info.max)) if info else (0.0, 1.0)
                    )
                    ch = {
                        "active": True,
                        "coefficient": 1.0,
                        "color": color,
                        "family": "linear",
                        "window": {
                            "min": float(win_min),
                            "max": float(win_max),
                            "start": float(win_min),
                            "end": float(win_max),
                        },
                        "label": label,
                    }
                    channel_meta.append(ch)

            # Write metadata
            self._write_metadata(image_name, axes_meta, channel_meta, creator_info)

        def _compute_levels(self, max_levels: Optional[int]) -> List[Tuple[int, ...]]:
            """
            Calculate all multiresolution level shapes by repeatedly scaling.
            Minimum dimension size will always be 1 and stops when no further reduction
            or at max_levels.

            Parameters
            ----------
            max_levels : Optional[int]
                Number of levels to compute (including base level).
                None for full descent.

            Returns
            -------
            List[Tuple[int, ...]]
                List of 5D shape tuples for each pyramid level.
            """
            shapes = [self.shape]
            lvl = 1
            while max_levels is None or lvl < max_levels:
                prev = shapes[-1]
                nxt = tuple(
                    max(1, prev[i] // self.scale_factors[i])
                    if i >= 2 and self.scale_factors[i] > 1
                    else prev[i]
                    for i in range(5)
                )
                if nxt == prev:
                    break
                shapes.append(nxt)
                lvl += 1
            return shapes

        def _prepare_parameter(
            self,
            param: Any,
            default_fn: Callable[[Tuple[int, ...]], Tuple[int, ...]],
            name: str,
            required: bool = True,
        ) -> List[Optional[Tuple[int, ...]]]:
            """
            Standardize chunk or shard specification across levels.

            Parameters
            ----------
            param : Union[str, Tuple[int, ...], List[Tuple[int, ...]], None]
                "auto", None, a tuple, or list of tuples per level.
            default_fn : Callable
                Function to generate default value for a level.
            name : str
                Parameter name ("chunks" or "shards").
            required : bool
                If False, None is allowed and yields None per level

            Returns
            -------
            List[Optional[Tuple[int, ...]]]
                List of parameter tuples (or None) per level.
            """

            # TODO: This function kinda sucks, needs better
            levels = len(self.level_shapes)

            # Auto-chunk only applies when name == "chunks"
            if param == "auto" and name == "chunks":
                return [default_fn(s) for s in self.level_shapes]

            if param is None:
                if not required:
                    return [None] * levels
                # repeat the default tuple for every level
                default = default_fn(self.level_shapes[0])
                return [default] * levels

            # Normalize into a list of length `levels`
            items: List[Tuple[int, ...]]
            if isinstance(param, list):
                items = param
            else:
                items = [param] * levels

            if len(items) != levels:
                raise ValueError(f"Length of {name} list must be {levels}")

            # Clamp each item to the level shape
            return [
                tuple(min(items[i][d], self.level_shapes[i][d]) for d in range(5))
                for i in range(levels)
            ]

        @staticmethod
        def _init_store(store: Union[str, zarr.storage.StoreLike]) -> zarr.Group:
            """
            Create or open a Zarr group at the given store location.

            Parameters
            ----------
            store : Union[str, zarr.storage.StoreLike]
                Path or Store object for output.

            Returns
            -------
            zarr.Group
                The root Zarr group for writing.
            """
            if (
                isinstance(store, str) and "://" in store
            ):  # TODO: make this a better check
                fs = zarr.storage.FsspecStore(store, mode="w")
                return zarr.group(store=fs, overwrite=True)
            return zarr.group(store=store, overwrite=True)

        def _create_arrays(self, resolver: Optional[BloscCodec]) -> List[zarr.Array]:
            """
            Create Zarr arrays for each multiscale level in the root group.

            Parameters
            ----------
            compressor : Optional[BloscCodec]
                Compressor codec to apply to chunks.

            Returns
            -------
            List[zarr.Array]
                Zarr array objects for each level.
            """
            comp = resolver or BloscCodec(
                cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
            )
            arrays: List[zarr.Array] = []
            for lvl, shape in enumerate(self.level_shapes):
                chunks_lvl = self.chunks[lvl]
                shards_lvl = self.shards[lvl]
                if chunks_lvl is not None and shards_lvl is not None:
                    # safe to zip two tuples
                    shards_param = tuple(c * s for c, s in zip(chunks_lvl, shards_lvl))
                else:
                    shards_param = None

                arr = self.root.create_array(
                    name=str(lvl),
                    shape=shape,
                    chunks=chunks_lvl,
                    shards=shards_param,
                    dtype=self.dtype,
                    compressors=comp,
                )
                arrays.append(arr)
            return arrays

        def _write_metadata(
            self,
            name: str,
            axes: List[dict],
            channels: Optional[List[dict]],
            creator: Optional[dict],
        ) -> None:
            """
            Write the 'ome' attribute on the root group with multiscale, OMERO,
            and creator metadata.

            Parameters
            ----------
            name : str
                Image name for metadata.
            axes : List[dict]
                Axes metadata list.
            channels : Optional[List[dict]]
                OMERO channel metadata list.
            creator : Optional[dict]
                Creator metadata dictionary.

            Returns
            -------
            None
            """
            multiscale: Dict[str, Any] = {
                "version": "0.5",
                "name": name,
                "axes": axes,
                "datasets": [
                    {
                        "path": str(i),
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [
                                    self.axes_scale[j]
                                    * (self.scale_factors[j] ** i if j >= 2 else 1)
                                    for j in range(len(self.shape))
                                ],
                            }
                        ],
                    }
                    for i in range(len(self.level_shapes))
                ],
            }
            ome: Dict[str, Any] = {"multiscales": [multiscale], "version": "0.5"}
            if channels:
                ome["omero"] = {"version": "0.5", "channels": channels}
            if creator:
                ome["_creator"] = creator
            self.root.attrs.update({"ome": ome})

        def _suggest_chunks(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
            """
            Suggest chunk shapes targeting ~64MB per chunk based on level shape.

            Parameters
            ----------
            shape : Tuple[int, ...]
                5D shape of the multiscale level.

            Returns
            -------
            Tuple[int, ...]
                Chunk sizes (t, c, z, y, x).
            """
            bpe = self.dtype.itemsize
            maxe = (64 << 20) // bpe
            base = int(math.sqrt(maxe))
            y = min(shape[3], base)
            x = min(shape[4], maxe // y) or 1
            z = min(shape[2], max(1, maxe // (y * x)))
            return (1, 1, z, y, x)

        def write_full_volume(self, data: np.ndarray) -> None:
            """
            Write an entire 5D image (T, C, Z, Y, X) into the Zarr store at all levels.

            Parameters
            ----------
            data : np.ndarray
                Full-resolution 5D image array.

            Returns
            -------
            None
            """
            assert data.shape == self.shape, "Input data shape must match base shape"
            self.datasets[0][:] = data
            cur = data
            for i in range(1, len(self.level_shapes)):
                dn = downsample_data(cur, self.scale_factors)
                self.datasets[i][:] = dn
                cur = dn

        def write_timepoint(self, t_index: int, data_t: np.ndarray) -> None:
            """
            Write a single timepoint (C, Z, Y, X) across all pyramid levels.

            Parameters
            ----------
            t_index : int
                Index of the timepoint to write.
            data_t : np.ndarray
                4D image array (C, Z, Y, X) for the given timepoint.

            Returns
            -------
            None
            """
            expected = (self.shape[1], self.shape[2], self.shape[3], self.shape[4])
            assert data_t.shape == expected, f"data_t must have shape {expected}"
            self.datasets[0][t_index, :, :, :, :] = data_t
            cur = data_t[np.newaxis, ...]
            for i in range(1, len(self.level_shapes)):
                dn = downsample_data(cur, self.scale_factors)
                self.datasets[i][t_index, :, :, :, :] = dn[0]
                cur = dn
