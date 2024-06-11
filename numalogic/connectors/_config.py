from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from numalogic.connectors.utils.aws.config import RDSConnectionConfig
from numalogic.connectors.exceptions import RDSFetcherConfValidationException


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
    agg: list[str] = field(default_factory=lambda: ["sum"])


@dataclass
class FilterConf:
    inclusion_filters: Optional[list[dict]] = None
    exclusion_filters: Optional[list[dict]] = None


@dataclass
class DruidFetcherConf:
    datasource: str
    static_filters: Optional[FilterConf] = None
    dimensions: list[str] = field(default_factory=list)
    aggregations: dict = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)
    pivot: Optional[Pivot] = field(default_factory=lambda: Pivot())
    granularity: str = "minute"

    def __post_init__(self):
        from pydruid.utils.aggregators import doublesum

        if not self.aggregations:
            self.aggregations = {"count": doublesum("count")}


@dataclass
class RDSFetcherConf:
    """
    RDSFetcherConf class represents the configuration for fetching data from an RDS data source.

    Args:
        datasource (str): The name of the data source.
        dimensions (list[str]): A list of dimension column names.
        group_by (list[str]): A list of column names to group the data by.
        pivot (Pivot): An instance of the Pivot class representing the pivot configuration.
        hash_query_type (bool): A boolean indicating whether to use hash query type.
        hash_column_name (Optional[str]): The name of the hash column. (default: None)
        datetime_column_name (str): The name of the datetime column. (default: "eventdatetime")
        metrics (list[str]): A list of metric column names.

    Methods
    -------
        __post_init__(): Performs post-initialization validation checks.

    Raises
    ------
        RDSFetcherConfValidationException: If the hash_query_type is enabled
        but hash_column_name is not provided.
    """

    datasource: str
    dimensions: list[str]
    # metric column names
    metrics: list[str]
    group_by: list[str] = field(default_factory=list)
    pivot: Optional[Pivot] = field(default_factory=lambda: Pivot())
    hash_query_type: bool = True
    hash_column_name: str = "model_md5_hash"
    datetime_column_name: str = "eventdatetime"

    def __post_init__(self):
        if self.hash_query_type:
            if self.hash_column_name.strip() == "":
                raise RDSFetcherConfValidationException(
                    "when hash_query_type is enabled, hash_column_name is required property "
                )


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
class RDSConf:
    """
    Class representing the configuration for fetching data from an RDS data source.

    Args:
        connection_conf (RDSConnectionConfig): An instance of the RDSConnectionConfig class
        representing the connection configuration.
        delay_hrs (float): The delay in hours for fetching data. Defaults to 3.0.
        fetcher (Optional[RDSFetcherConf]): An optional instance of the RDSFetcherConf class
            representing the fetcher configuration. Defaults to None.
        id_fetcher (Optional[dict[str, RDSFetcherConf]]): An optional dictionary mapping IDs to
            instances of the RDSFetcherConf class representing the fetcher configuration.
            Defaults to None.
    """

    connection_conf: RDSConnectionConfig
    delay_hrs: float = 3.0
    fetcher: Optional[RDSFetcherConf] = None
    id_fetcher: Optional[dict[str, RDSFetcherConf]] = None
