from importlib.util import find_spec

from numalogic.connectors._config import (
    RedisConf,
    PrometheusConf,
    ConnectorConf,
    DruidConf,
    DruidFetcherConf,
    ConnectorType,
    RDSConf,
    RDSFetcherConf,
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
    "RDSFetcher",
    "RDSConf",
    "RDSFetcherConf",
]

if find_spec("boto3"):
    from numalogic.connectors.rds import RDSFetcher  # noqa: F401

    __all__.append("RDSFetcher")

if find_spec("pydruid"):
    from numalogic.connectors.druid import DruidFetcher  # noqa: F401

    __all__.append("DruidFetcher")
