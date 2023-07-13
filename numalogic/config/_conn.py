from dataclasses import dataclass, field


@dataclass
class PrometheusConf:
    server: str
    pushgateway: str


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
    from pydruid.utils.aggregators import doublesum

    datasource: str
    dimensions: list[str] = field(default_factory=list)
    aggregations: dict = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)
    pivot: Pivot = field(default_factory=lambda: Pivot())
    granularity: str = "minute"
    hours: float = 36

    def __post_init__(self):
        if not self.aggregations:
            self.aggregations = {"count": self.doublesum("count")}
