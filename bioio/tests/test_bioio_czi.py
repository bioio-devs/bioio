# Integration tests for bioio + bioio-czi.
# bioio-czi is not directly referenced here, but installed in the test dependency
# group in pyproject.toml.
# Tests in the bioio_czi repository do not use bioio (just bioio-base).
from bioio import BioImage


def test_bioimage_can_ignore_query_strings() -> None:
    # We can open a file over the internet with a path that has a query string.
    BioImage(
        "https://allencell.s3.amazonaws.com/aics/hipsc_12x_overview_image_dataset/"
        "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
        "?versionId=_KYMRhRvKxnu727ssMD2_fZD5CmQMNw6"
    )
