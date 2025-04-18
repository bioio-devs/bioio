# Integration tests for bioio + bioio-czi.
# bioio-czi is not directly referenced here, but installed in the test dependency
# group in pyproject.toml.
# Tests in the bioio_czi repository do not use bioio (just bioio-base).
#
# Copyright (C) 2025 Allen Institute
#
# The following copyright license applies to THIS FILE ONLY, not the rest of bioio.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from bioio import BioImage


def test_bioimage_can_ignore_query_strings() -> None:
    # We can open a file over the internet with a path that has a query string.
    BioImage(
        "https://allencell.s3.amazonaws.com/aics/hipsc_12x_overview_image_dataset/"
        "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
        "?versionId=_KYMRhRvKxnu727ssMD2_fZD5CmQMNw6"
    )
