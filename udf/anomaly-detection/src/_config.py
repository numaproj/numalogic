from enum import Enum
from typing import List
from omegaconf import MISSING
from dataclasses import dataclass, field

from numalogic.config import NumalogicConf

from src.connectors import RedisConf, RegistryConf, PrometheusConf
from src.connectors._config import DruidConf, DruidFetcherConf


@dataclass
class UnifiedConf:
    strategy: str = "max"
    weights: List[float] = field(default_factory=list)


@dataclass
class ReTrainConf:
    train_hours: int = 24 * 8  # 8 days worth of data
    min_train_size: int = 2000
    retrain_freq_hr: int = 24
    resume_training: bool = False
    model_expiry_sec: int = 86400  # 24 hrs
    dedup_expiry_sec: int = 1800  # 30 days


@dataclass
class StaticThreshold:
    upper_limit: List[int] = field(default_factory=list)
    weight: List[float] = field(default_factory=list)


class DataSource(str, Enum):
    PROMETHEUS = "prometheus"
    DRUID = "druid"


@dataclass
class StreamConf:
    config_id: str = "default"
    source: str = DataSource.PROMETHEUS.value
    window_size: int = 12
    composite_keys: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    retrain_conf: ReTrainConf = field(default_factory=lambda: ReTrainConf())
    static_threshold: StaticThreshold = field(default_factory=lambda: StaticThreshold())
    numalogic_conf: NumalogicConf = MISSING
    unified_config: UnifiedConf = field(default_factory=lambda: UnifiedConf())
    druid_fetcher: DruidFetcherConf = MISSING


@dataclass
class Configs:
    configs: List[StreamConf]


@dataclass
class PipelineConf:
    redis_conf: RedisConf
    registry_conf: RegistryConf
    prometheus_conf: PrometheusConf = MISSING
    druid_conf: DruidConf = MISSING
