from typing import ClassVar

from numalogic.config.factory import _ObjectFactory
from numalogic.tools.exceptions import UnknownConfigArgsError


class ConnectorFactory(_ObjectFactory):
    """Factory class for data test_connectors."""

    _CLS_SET: ClassVar[frozenset] = frozenset(
        {"PrometheusFetcher", "DruidFetcher", "RDSFetcher"}
    )

    @classmethod
    def get_cls(cls, name: str):
        import nlconnectors as conn

        try:
            return getattr(conn, name)
        except AttributeError as err:
            if name in cls._CLS_SET:
                raise ImportError(
                    "Please install the required dependencies for the connector you want to use."
                ) from err
            raise UnknownConfigArgsError(
                f"Invalid name provided for ConnectorFactory: {name}"
            ) from None
