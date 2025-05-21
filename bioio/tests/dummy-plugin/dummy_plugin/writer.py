from typing import Any

import bioio_base as biob
from bioio_base.writer import Writer


class DummyWriter(Writer):
    """
    A dummy writer for testing bioio.writers entry-point discovery.
    """

    @staticmethod
    def save(
        data: biob.types.ArrayLike,
        uri: biob.types.PathLike,
        dim_order: str = biob.dimensions.DEFAULT_DIMENSION_ORDER,
        **kwargs: Any,
    ) -> None:
        """
        Stub implementation that deliberately isn’t implemented.
        """
        raise NotImplementedError("DummyWriter.save is a stub")
