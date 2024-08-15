#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import sys

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint
else:
    from importlib.metadata import EntryPoint

import bioio_base as biob
import numpy as np
import pytest
from bioio_base import exceptions
from bioio_base.reader import Reader
from bioio_base.reader_metadata import ReaderMetadata
from bioio_base.types import ImageLike

from bioio import BioImage
from bioio.array_like_reader import ArrayLikeReader


def test_bioimage_with_text_file(sample_text_file: pathlib.Path) -> None:
    with pytest.raises(biob.exceptions.UnsupportedFileFormatError):
        BioImage(sample_text_file)


def test_bioimage_with_missing_file(tmp_path: pathlib.Path) -> None:
    # Construct full filepath
    uri = tmp_path / "does-not-exist-klafjjksdafkjl.bad"
    with pytest.raises(biob.exceptions.UnsupportedFileFormatError):
        BioImage(uri)


def test_bioimage_determine_arraylike() -> None:
    # Arrage
    test_image = np.random.rand(10, 10)

    # Act
    result = BioImage.determine_plugin(test_image)

    # Assert
    assert isinstance(result.entrypoint, EntryPoint)
    assert isinstance(result.metadata, ReaderMetadata)
    assert isinstance(result.timestamp, float)
    assert issubclass(result.metadata.get_reader(), Reader)
    assert result.metadata.get_reader() is ArrayLikeReader


@pytest.mark.parametrize(
    "image, reader_class, expected_exception",
    [
        (
            "s3://my_bucket/my_file.tiff",
            ArrayLikeReader,
            exceptions.UnsupportedFileFormatError,
        ),
        (np.random.rand(10, 10), Reader, exceptions.UnsupportedFileFormatError),
        (
            ["s3://my_bucket/my_file.tiff", "s3://my_bucket/my_file.tiff"],
            ArrayLikeReader,
            exceptions.UnsupportedFileFormatError,
        ),
        (
            [np.random.rand(10, 10), np.random.rand(10, 10)],
            Reader,
            exceptions.UnsupportedFileFormatError,
        ),
    ],
)
def test_bioimage_submission_data_reader_type_alignment(
    image: ImageLike, reader_class: Reader, expected_exception: Exception
) -> None:
    with pytest.raises(expected_exception):
        BioImage(image, reader=reader_class)


def test_bioimage_attempts_s3_read_with_anon_attr(
    sample_text_file: pathlib.Path, dummy_plugin: str
) -> None:
    # Arrange
    from dummy_plugin import Reader as DummyReader

    err_msg = "anon is not True"

    class FakeReader(DummyReader):
        def __init__(self, _: ImageLike, fs_kwargs: dict = {}):
            if fs_kwargs.get("anon", False) is not True:
                raise biob.exceptions.UnsupportedFileFormatError(
                    "test", "test", err_msg
                )

    # Act / Assert
    BioImage("s3://this/could/go/anywhere.ome.zarr", reader=FakeReader)

    # (sanity-check) Make sure it would have failed if not an S3 URI
    # that gets given the anon attribute
    with pytest.raises(biob.exceptions.UnsupportedFileFormatError) as err:
        BioImage(sample_text_file, reader=FakeReader)

    assert err_msg in str(err.value)
