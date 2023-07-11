from dataclasses import dataclass, field

from pydruid.utils.aggregators import doublesum


@dataclass
class PrometheusConf:
    server: str
    pushgateway: str


@dataclass
class RegistryConf:
    tracking_uri: str


@dataclass
class RedisConf:
    host: str
    port: int
    expiry: int = 300
    master_name: str = "mymaster"


@dataclass
class DruidConf:
    url: str
    endpoint: str


@dataclass
class DruidFetcherConf:
    datasource: str
    dimensions: list[str] = field(default_factory=list)
    aggregations: dict = field(default_factory=lambda: {"count": doublesum("count")})
    group_by: list[str] = field(default_factory=list)
    pivot: dict = field(default_factory=dict)
    granularity: str = "minute"
    hours: float = 24
