import pathlib
from importlib import import_module
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
from bioio.tests.conftest import TestPluginSpec
from bioio.tests.helpers.mock_reader import PluginFactoryFixture


def test_bioimage_with_text_file(sample_text_file: pathlib.Path) -> None:
    with pytest.raises(biob.exceptions.UnsupportedFileFormatError):
        BioImage(sample_text_file)


def test_bioimage_with_missing_file(tmp_path: pathlib.Path) -> None:
    # Construct full filepath
    uri = tmp_path / "does-not-exist-klafjjksdafkjl.bad"
    with pytest.raises(biob.exceptions.UnsupportedFileFormatError):
        BioImage(uri)


def test_bioimage_determine_arraylike() -> None:
    # Arrange
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
    plugin_factory: PluginFactoryFixture,
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
    plugin_factory: PluginFactoryFixture,
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


@pytest.mark.parametrize(
    "specs, reader_module_names, expected_winner_idx",
    [
        # Case 1: single explicit reader overrides extensions
        (
            [
                TestPluginSpec(
                    name="single_reader_plugin",
                    supported_extensions=[".txt"],  # does not match ".weird"
                    fail_on_is_supported=False,
                )
            ],
            ["single_reader_plugin"],
            0,
        ),
        # Case 2: explicit reader list also overrides extensions (first wins)
        (
            [
                TestPluginSpec(
                    name="list_reader_plugin_a",
                    supported_extensions=[".txt"],
                    fail_on_is_supported=False,
                ),
                TestPluginSpec(
                    name="list_reader_plugin_b",
                    supported_extensions=[".txt"],
                    fail_on_is_supported=False,
                ),
            ],
            ["list_reader_plugin_a", "list_reader_plugin_b"],
            0,
        ),
        # Case 3: Second explicit reader supports file (dif ext)
        (
            [
                TestPluginSpec(
                    name="wrong_ext",
                    supported_extensions=[".tiff"],
                    fail_on_is_supported=True,
                ),
                TestPluginSpec(
                    name="supported",
                    supported_extensions=[".txt"],
                    fail_on_is_supported=False,
                ),
            ],
            ["wrong_ext", "supported"],
            1,
        ),
        # Case 4: Second explicit reader supports file (same ext)
        (
            [
                TestPluginSpec(
                    name="first_unsupported",
                    supported_extensions=[".foo"],
                    fail_on_is_supported=True,
                    fail_message="unsupported - first",
                ),
                TestPluginSpec(
                    name="second_supported",
                    supported_extensions=[".foo"],
                    fail_on_is_supported=False,
                ),
            ],
            ["first_unsupported", "second_supported"],
            1,
        ),
    ],
)
def test_bioimage_explicit_readers_priority(
    plugin_factory: PluginFactoryFixture,
    specs: list[TestPluginSpec],
    reader_module_names: list[str],
    expected_winner_idx: int,
) -> None:
    """
    If a reader (or reader list) is explicitly provided to BioImage, extension
    matching and plugin discovery are bypassed. BioImage tries ONLY the provided
    readers, in order, and uses the first one that constructs successfully.
    """
    # Arrange
    plugin_factory(specs)

    readers = []
    for name in reader_module_names:
        mod = import_module(f"bioio_test_plugins.{name}")
        readers.append(getattr(mod, "Reader"))

    reader_arg = readers[0] if len(readers) == 1 else readers

    # Act
    img = BioImage("image.weird", reader=reader_arg)

    # Assert
    assert isinstance(img.reader, readers[expected_winner_idx])


def test_bioimage_reader_list_aggregates_failures_when_all_fail(
    plugin_factory: PluginFactoryFixture,
) -> None:
    """
    If all explicitly provided readers fail initialization,
    BioImage raises and includes the per-reader failure reasons.
    """
    # Arrange
    plugin_factory(
        [
            TestPluginSpec(
                name="fails_a",
                supported_extensions=[".foo"],
                fail_on_is_supported=True,
                fail_message="unsupported - A",
            ),
            TestPluginSpec(
                name="fails_b",
                supported_extensions=[".foo"],
                fail_on_is_supported=True,
                fail_message="unsupported - B",
            ),
        ]
    )

    readers = []
    for name in ["fails_a", "fails_b"]:
        mod = import_module(f"bioio_test_plugins.{name}")
        readers.append(getattr(mod, "Reader"))

    # Act / Assert
    with pytest.raises(biob.exceptions.UnsupportedFileFormatError) as err:
        BioImage("image.foo", reader=readers)

    # Assert
    msg = str(err.value)
    assert "unsupported - A" in msg
    assert "unsupported - B" in msg
