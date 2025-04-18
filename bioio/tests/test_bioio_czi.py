from bioio import BioImage
from bioio.plugins import plugin_feasibility_report


def test_bioimage_can_ignore_query_strings():
    # Arrange
    uri = (
        "https://allencell.s3.amazonaws.com/aics/hipsc_12x_overview_image_dataset/"
        "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
        "?versionId=_KYMRhRvKxnu727ssMD2_fZD5CmQMNw6"
    )

    # Act
    image = BioImage(uri)

    # Assert