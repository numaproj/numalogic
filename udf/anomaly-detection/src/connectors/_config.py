from dataclasses import dataclass, field

from pydruid.utils.aggregators import doublesum


@dataclass
class PrometheusConf:
    server: str
    pushgateway: str
    scrape_interval: int = 30


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
