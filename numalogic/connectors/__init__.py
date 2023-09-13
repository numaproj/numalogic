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

try:
    find_spec("pydruid")
except (ImportError, ModuleNotFoundError):
    pass
else:

    __all__.append("DruidFetcher")
