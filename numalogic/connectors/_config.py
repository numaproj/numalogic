from dataclasses import dataclass, field
from enum import Enum, IntEnum

from omegaconf import MISSING
from pydruid.utils.aggregators import doublesum


class ConnectorType(IntEnum):
    REDIS = 0
    PROMETHEUS = 1
    DRUID = 2


@dataclass
class ConnectorConf:
    url: str


@dataclass
class PrometheusConf(ConnectorConf):
    pushgateway: str
    scrape_interval: int = 30


@dataclass
class RedisConf(ConnectorConf):
    port: int
    expiry: int = 300
    master_name: str = "mymaster"


@dataclass
class Pivot:
    index: str = "timestamp"
    columns: list[str] = field(default_factory=list)
    value: list[str] = field(default_factory=lambda: ["count"])


@dataclass
class DruidFetcherConf:
    datasource: str
    dimensions: list[str] = field(default_factory=list)
    aggregations: dict = field(default_factory=lambda: {"count": doublesum("count")})
    group_by: list[str] = field(default_factory=list)
    pivot: Pivot = field(default_factory=lambda: Pivot())
    granularity: str = "minute"


@dataclass
class DruidConf(ConnectorConf):
    endpoint: str
    fetcher: DruidFetcherConf = MISSING
