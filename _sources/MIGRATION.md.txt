# Migration from AICSImageIO v4 to BioIO v1

## How different is the API?

Very little about the API has changed between `aicsimageio` version 4.0.0+ and `bioio`.  The API is largely the same with the biggest differences being:
* You use a different Python package (`bioio` instead of `aicsimageio`)
    * **Important** - See the [Reader Installation Instructions](https://github.com/bioio-devs/bioio/main/README.md#reader-installation) for which additional packages (`bioio` plug-ins) you'll need to support the files you want to interact with
* Reader plugins are separate packages that are installed separately.  OmeTiffReader and OmeZarrReader are preinstalled with the default distribution of bioio.
* The main class exported was renamed from `AICSImage` to `BioImage`

Example of **OLD** code using `aicsimageio`
```python
from aicsimageio import AICSImage

image = AICSImage("/some/path/to/my/path")
print(image.dims)
print(image.scenes)
```

Example of how that looks using `bioio`

```python
from bioio import BioImage

image = BioImage("/some/path/to/my/path")
print(image.dims)
print(image.scenes)
```

## Why use BioIO rather than AICSImageIO?
A few reasons:
* Licensing is easier to understand and manage with `bioio` since each file reader is installed separately.
* Fewer dependencies are necessary for users that require only a subset of the total available readers.
* Readers can evolve independently of each other. For example (totally hypothetical), if `bioio-czi` (AKA the CZI reader) needs to start using `dask` version `4.0.0+`, but `bioio-tifffile` (AKA the TIFF reader) needs `dask` version `3.4.0-3.9.0` then we can upgrade `bioio-czi` without conflicting with `bioio-tifffile`.
* Community Reader development is now made easier.  You can write and maintain 
your own custom reader without having to submit it to the core bioio package.  We always encourage publicizing useful readers for the imaging community at large.
* `aicsimageio` will be moving into "maintenance" only mode, meaning only critical bugfixes will be made to the codebase.  `bioio` will be where new features / support for new version of Python are made.

