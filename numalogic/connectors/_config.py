from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class ConnectorType(IntEnum):
    redis = 0
    prometheus = 1
    druid = 2


@dataclass
class ConnectorConf:
    url: str


@dataclass
class PrometheusConf(ConnectorConf):
    pushgateway: str = ""
    scrape_interval: int = 30
    return_labels: list[str] = field(default_factory=list)


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
    aggregations: dict = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)
    pivot: Pivot = field(default_factory=lambda: Pivot())
    granularity: str = "minute"

    def __post_init__(self):
        from pydruid.utils.aggregators import doublesum

        if not self.aggregations:
            self.aggregations = {"count": doublesum("count")}


@dataclass
class DruidConf(ConnectorConf):
    """
    Class for configuring Druid connector.

    Args:
        endpoint: Druid endpoint
        delay_hrs: Delay in hours for fetching data from Druid
        fetcher: DruidFetcherConf
        id_fetcher: dict of DruidFetcherConf for fetching ids

    Note: Either one of the fetcher or id_fetcher should be provided.
    """

    endpoint: str
    delay_hrs: float = 3.0
    fetcher: Optional[DruidFetcherConf] = None
    id_fetcher: Optional[dict[str, DruidFetcherConf]] = None
