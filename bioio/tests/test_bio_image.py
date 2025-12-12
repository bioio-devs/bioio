import pathlib
from importlib.metadata import EntryPoint
from typing import Callable, Iterable

import bioio_base as biob
import numpy as np
import pytest
from bioio_base import exceptions
from bioio_base.reader import Reader
from bioio_base.reader_metadata import ReaderMetadata
from bioio_base.types import ImageLike

from bioio import BioImage
from bioio.array_like_reader import ArrayLikeReader
from bioio.tests.conftest import TestPluginSpec


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
    sample_text_file: pathlib.Path,
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
) -> None:
    # Arrange: Fake s3 plugin Reader.__init__ checks fs_kwargs["anon"]
    err_msg = "anon is not True"
    specs = [
        TestPluginSpec(
            name="anon_check_plugin",
            supported_extensions=[".ome.zarr"],
            check_anon_in_init=True,
            fail_message=err_msg,
        )
    ]
    plugin_factory(specs)

    # Import to specify fake reader
    from bioio_test_plugins.anon_check_plugin import Reader as AnonCheckReader

    # Act/Assert 1: For S3 URIs, BioImage should pass anon=True
    BioImage("s3://this/could/go/anywhere.ome.zarr", reader=AnonCheckReader)

    # Act/Assert 2: for a local path, anon should NOT be forced.
    with pytest.raises(biob.exceptions.UnsupportedFileFormatError) as err:
        BioImage(sample_text_file, reader=AnonCheckReader)

    assert err_msg in str(err.value)


def test_bioimage_can_ignore_query_strings(
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]]
) -> None:
    # Arrange: Fake czi plugin
    specs = [
        TestPluginSpec(
            name="accepting_plugin",
            supported_extensions=[".czi"],
        )
    ]
    plugin_factory(specs)

    # Act / Assert:
    # The accepting plugin claims to support .czi files, and it will not error
    # when instantiating a reader. The following should succeed if BioImage
    # correctly ignores the query string when evaluating plugin support.
    BioImage(
        "https://allencell.s3.amazonaws.com/aics/hipsc_12x_overview_image_dataset/"
        "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
        "?versionId=_KYMRhRvKxnu727ssMD2_fZD5CmQMNw6"
    )


def test_bioimage_reader_priority_usage(
    tmp_path: pathlib.Path,
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
) -> None:
    # Arrange: two plugins that both claim support for the same extension
    specs = [
        TestPluginSpec(
            name="priority_first",
            supported_extensions=[".foo"],
            fail_on_is_supported=False,
        ),
        TestPluginSpec(
            name="priority_second",
            supported_extensions=[".foo"],
            fail_on_is_supported=False,
        ),
    ]
    plugin_factory(specs)

    img_path = tmp_path / "image.foo"
    img_path.write_text("dummy")

    # Import the synthetic Reader classes from our ephemeral modules
    from bioio_test_plugins.priority_first import Reader as FirstReader
    from bioio_test_plugins.priority_second import Reader as SecondReader

    # Act: pass a *priority list* of readers.
    img = BioImage(str(img_path), reader=[SecondReader, FirstReader])

    # Assert: Correct Reader
    assert isinstance(img.reader, SecondReader)


def test_bioimage_reader_priority_overrides_default_order(
    tmp_path: pathlib.Path,
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
) -> None:
    # Arrange: two plugins for the SAME extension, but with different
    # "specificity" according to bioio.plugins.get_plugins()
    specs = [
        TestPluginSpec(
            name="specific_plugin",
            supported_extensions=[".ome.tiff", ".tiff"],
            fail_on_is_supported=False,
        ),
        TestPluginSpec(
            name="generic_plugin",
            supported_extensions=[".ome.tiff", ".tiff", ".ome.tif", ".tif", ".png"],
            fail_on_is_supported=False,
        ),
    ]
    plugin_factory(specs)

    img_path = tmp_path / "image.ome.tiff"
    img_path.write_text("dummy")

    # Import the synthetic Reader classes
    from bioio_test_plugins.generic_plugin import Reader as GenericReader
    from bioio_test_plugins.specific_plugin import Reader as SpecificReader

    # Act 1: Verify default
    default_img = BioImage(str(img_path))

    # Act 2: Override
    priority_img = BioImage(
        str(img_path),
        reader=[GenericReader, SpecificReader],
    )

    # Assert 1: Correct default
    assert isinstance(default_img.reader, SpecificReader)

    # Assert 2: Priority Override
    assert isinstance(priority_img.reader, GenericReader)


def test_bioimage_single_reader_overrides_plugins_and_extensions(
    tmp_path: pathlib.Path,
    plugin_factory: Callable[[Iterable[TestPluginSpec]], list[EntryPoint]],
) -> None:
    # Arrange: create a synthetic plugin whose supported_extensions DO NOT
    # match the file we are about to read.
    specs = [
        TestPluginSpec(
            name="single_reader_plugin",
            supported_extensions=[".txt"],
            fail_on_is_supported=False,
        )
    ]
    plugin_factory(specs)

    img_path = tmp_path / "image.weird"
    img_path.write_text("dummy")

    # Import the synthetic Reader class created by the plugin spec
    from bioio_test_plugins.single_reader_plugin import Reader as SingleReader

    # Act: pass single Reader
    img = BioImage(str(img_path), reader=SingleReader)

    # Assert: the forced reader was used
    assert isinstance(img.reader, SingleReader)

    # And plugin discovery was bypassed (no PluginEntry stored)
    assert img._plugin is None  # type: ignore[attr-defined]
