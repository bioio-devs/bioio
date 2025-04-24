from typing import List

import bioio_base.reader_metadata


class ReaderMetadata(bioio_base.reader_metadata.ReaderMetadata):
    """
    Notes
    -----
    Defines metadata for the reader itself (not the image read),
    such as supported file extensions.
    """

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        Return a list of file extensions this plugin supports reading.
        """
        # Feel free to add any extensions here. This plugin is intended to pretend to
        # read anything.
        return [
            ".tiff",
            ".tif",
            ".ome.tiff",
            ".ome.tif",
            ".zarr",
            ".ome.zarr",
            ".czi",
            ".nd2",
        ]

    @staticmethod
    def get_reader() -> bioio_base.reader.Reader:
        """
        Return the reader this plugin represents
        """
        from .reader import Reader

        return Reader
