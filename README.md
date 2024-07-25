# BioIO

[![Build Status](https://github.com/bioio-devs/bioio/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio/actions)
[![Documentation](https://github.com/bioio-devs/bioio/actions/workflows/docs.yml/badge.svg)](https://bioio-devs.github.io/bioio)
[![PyPI version](https://badge.fury.io/py/bioio.svg)](https://badge.fury.io/py/bioio)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.9+](https://img.shields.io/badge/python-3.9,3.10,3.11-blue.svg)](https://www.python.org/downloads/release/python-390/)

Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python

---

## Documentation

[See the full documentation on our GitHub pages site](https://bioio-devs.github.io/bioio/OVERVIEW.html)

## Example Usage (see full documentation for more examples)

Install bioio alongside OME TIFF and OME ZARR plug-ins with pip (this example won't use the OME ZARR plug-in):

`pip install bioio bioio-ome-tiff bioio-ome-zarr`

```python
from bioio import BioImage

# Get a BioImage object
img = BioImage("my_file.tiff")  # selects the first scene found
img.data  # returns 5D TCZYX numpy array
img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
img.dims  # returns a Dimensions object
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

# Get the id of the current operating scene
img.current_scene

# Get a list valid scene ids
img.scenes

# Change scene using name
img.set_scene("Image:1")
# Or by scene index
img.set_scene(1)

# Use the same operations on a different scene
# ...
```
## Plug-in Registry

Bioio handles a variety of different image types through specific plug-ins. The bioio-dev supported plug-ins can be found within
this registry.


| Plug-in                | Extension   | Repository          |
|------------------------|-------------|---------------------|
| arraylike              | [ArrayLike](https://github.com/bioio-devs/bioio-base/blob/9ff0a17a20d09b1b11639d149b1e71801c9d68d8/bioio_base/types.py#L15)  | Built-In           |
| bioio-czi              | .czi        | [Repo](https://github.com/bioio-devs/bioio-czi)           |
| bioio-dv               | .dv, .r3d   | [Repo](https://github.com/bioio-devs/bioio-dv)           |
| bioio-imageio          | .jpg, .png, [Full List](https://github.com/bioio-devs/bioio-imageio/blob/6829370644b9780cfde35fa9d2cd5cea9f743681/bioio_imageio/reader_metadata.py#L26)  | [Repo](https://github.com/bioio-devs/bioio-imageio)           |
| bioio-lif              | .lif        | [Repo](https://github.com/bioio-devs/bioio-lif)           |
| bioio-nd2              | .nd2        | [Repo](https://github.com/bioio-devs/bioio-nd2)           |
| bioio-ome-tiff         | .ome.tiff, .tiff  | [Repo](https://github.com/bioio-devs/bioio-ome-tiff)           |
| bioio-ome-tiled-tiff   | .tiles.ome.tif   | [Repo](https://github.com/bioio-devs/bioio-ome-tiled-tiff)           |
| bioio-ome-zarr         | .zarr       | [Repo](https://github.com/bioio-devs/bioio-ome-zarr)           |
| bioio-sldy             | .sldy, .dir | [Repo](https://github.com/bioio-devs/bioio-sldy)           |
| bioio-tifffile         | .tif , .tiff| [Repo](https://github.com/bioio-devs/bioio-tifffile)           |
| bioio-tiff-glob        | .tiff (glob)| [Repo](https://github.com/bioio-devs/bioio-tiff-glob)           |
| bioio-bioformats       | [Full List](https://github.com/bioio-devs/bioio-bioformats/blob/175399d10d64194adcc7a6048c7b7537591824de/bioio_bioformats/reader_metadata.py#L24) | [Repo](https://github.com/bioio-devs/bioio-bioformats)           |

Each reader plugin should closely follow the specification laid out in `bioio-base`. As such, it is likely common that reader plugins won't distribute their own documentation and users should instead review  [`bioio_base.reader.Reader`](https://bioio-devs.github.io/bioio-base/bioio_base.html#bioio_base.reader.Reader) for API documentation for the underlying Reader API. We encourage plugin authors to publish their own documentation if they change or include new features into their published image readers.

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch)