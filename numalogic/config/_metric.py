from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING

from numalogic.config._conn import DruidFetcherConf, RedisConf, DruidConf, PrometheusConf
from numalogic.config._numa import NumalogicConf


@dataclass
class UnifiedConf:
    strategy: str = "max"
    weights: list[float] = field(default_factory=list)


@dataclass
class ReTrainConf:
    train_hours: int = 36
    min_train_size: int = 2000
    retrain_freq_hr: int = 8
    resume_training: bool = False


@dataclass
class StaticThresholdConf:
    upper_limit: int = 3
    weight: float = 0.0


@dataclass
class MetricConf:
    metric: str
    retrain_conf: ReTrainConf = field(default_factory=lambda: ReTrainConf())
    static_threshold: StaticThresholdConf = field(default_factory=lambda: StaticThresholdConf())
    numalogic_conf: NumalogicConf = MISSING


class DataSource(str, Enum):
    PROMETHEUS = "prometheus"
    DRUID = "druid"


@dataclass
class DataStreamConf:
    name: str = "default"
    source: str = DataSource.PROMETHEUS.value
    window_size: int = 12
    composite_keys: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    metric_configs: list[MetricConf] = field(default_factory=list)
    unified_config: UnifiedConf = field(default_factory=lambda: UnifiedConf())
    druid_fetcher: DruidFetcherConf = MISSING


@dataclass
class Configs:
    configs: list[DataStreamConf]


@dataclass
class PipelineConf:
    redis_conf: RedisConf
    # registry_conf: RegistryConf
    prometheus_conf: PrometheusConf = MISSING
    druid_conf: DruidConf = MISSING
