from importlib.util import find_spec

from numalogic.connectors._config import (
    RedisConf,
    PrometheusConf,
    ConnectorConf,
    DruidConf,
    DruidFetcherConf,
    ConnectorType,
)
from numalogic.connectors.prometheus import PrometheusFetcher

__all__ = [
    "RedisConf",
    "PrometheusConf",
    "ConnectorConf",
    "DruidConf",
    "DruidFetcherConf",
    "ConnectorType",
    "PrometheusFetcher",
]


if find_spec("pydruid"):
    from numalogic.connectors.druid import DruidFetcher  # noqa: F401

    __all__.append("DruidFetcher")
