from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from numalogic.connectors.utils.aws.config import RDSConfig


class ConnectorType(IntEnum):
    redis = 0
    prometheus = 1
    druid = 2
    rds = 3


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
class RDSFetcherConf:
    datasource: str
    dimensions: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    pivot: Pivot = field(default_factory=lambda: Pivot())
    hash_query_type: bool = field(default=True)
    hash_column_name: Optional[str] = field(default_factory=str)
    datetime_column_name: str = "eventdatetime"
    # metric column names
    metrics: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.hash_query_type:
            if not self.hash_column_name or self.hash_column_name.strip() == "":
                raise RDSFetcherConfValidationException(
                    "when hash_query_type is enabled, hash_column_name is required property ")


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


@dataclass
class RDSConf():
    connection_conf: RDSConfig
    delay_hrs: float = 3.0
    fetcher: Optional[RDSFetcherConf] = None
    id_fetcher: Optional[dict[str, RDSFetcherConf]] = None


if __name__ == "__main__":
    from numalogic.udfs._config import load_pipeline_conf

    config = load_pipeline_conf(
        "/Users/skondakindi/Desktop/codebase/ml/numalogic/tests/resources/rds_trainer_config.yaml")
    print(config)
