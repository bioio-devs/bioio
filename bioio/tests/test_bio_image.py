#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bioio_base as biob
import pytest

from bioio import BioImage

from .conftest import get_resource_full_path


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param(
            "example.txt",
            marks=pytest.mark.xfail(raises=biob.exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "does-not-exist-klafjjksdafkjl.bad",
            marks=pytest.mark.xfail(raises=FileNotFoundError),
        ),
    ],
)
def test_bioimage(
    filename: str,
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename)
    BioImage(uri)
