import numpy as np
import zarr
from ngff_zarr.validate import validate

from bioio.writers import OMEZarrWriter  # wherever you saved the class

output = "/Users/brian.whitney/Desktop/Repos/bioio/bioio/writers/image.ome.zarr"

# TODO: Delete this file, this is just for development


def create_zarr() -> None:
    # Synthetic test data
    T, C, Z, Y, X = 10, 3, 64, 1024, 1024
    img_data = np.random.randint(0, 65535, size=(T, C, Z, Y, X), dtype=np.uint16)

    # Instantiate writer: axes and OMERO channels are auto‐built under the hood
    writer = OMEZarrWriter(
        store="image.ome.zarr",  # output directory
        shape=img_data.shape,  # (10,3,64,1024,1024)
        dtype=img_data.dtype,  # np.uint16
        # You can override the defaults here if you like:
        axes_names=["t", "c", "z", "y", "x"],
        axes_types=["time", "channel", "space", "space", "space"],
        axes_units=[None, None, "µm", "µm", "µm"],
        axes_scale=[1.0, 1.0, 1.0, 0.5, 0.5],  # physical sizes
        scale_factors=(1, 1, 2, 2, 2),  # no downsample on T/C, 2× on Z/Y/X
        num_levels=4,  # levels 0→3
        chunks="auto",  # auto‐chunk ~64 MB
        shards=(1, 1, 4, 4, 4),  # optional
        channel_names=["DAPI", "FITC", "TRITC"],  # just names
        channel_colors=["0000FF", "00FF00", "FF0000"],  # just hex colors
        creator_info={"name": "My Writer", "version": "1.0.0"},
    )

    # Write everything
    writer.write_full_volume(img_data)

    group = zarr.open_group(output, mode="r")

    # Pull the OME-NGFF metadata directly from the 'ome' attribute
    ome_meta = group.attrs.asdict()
    # Validate against the NGFF 0.5 image schema
    validate(
        ome_meta,  # the dict to check
        version="0.5",  # NGFF spec version → spec/0.5/schemas/image.schema
        model="image",  # validate as the Image model
        strict=False,  # set True if you want to error on warnings
    )

    print("✅ OME-Zarr v0.5 metadata is valid!")
