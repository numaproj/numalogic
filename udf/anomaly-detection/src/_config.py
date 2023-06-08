from typing import List
from omegaconf import MISSING
from dataclasses import dataclass, field

from numalogic.config import NumalogicConf

from src.connectors import RedisConf, RegistryConf, PrometheusConf


@dataclass
class UnifiedConf:
    unified_metric_name: str = "default"
    unified_strategy: str = "max"
    unified_metrics: List[str] = field(default_factory=list)
    unified_weights: List[float] = field(default_factory=list)


@dataclass
class ReTrainConf:
    train_hours: int = 36
    min_train_size: int = 2000
    retrain_freq_hr: int = 8
    resume_training: bool = False


@dataclass
class StaticThreshold:
    upper_limit: int = 3
    weight: float = 0.0


@dataclass
class MetricConf:
    metric: str = "default"
    retrain_conf: ReTrainConf = field(default_factory=lambda: ReTrainConf())
    static_threshold: StaticThreshold = field(default_factory=lambda: StaticThreshold())
    numalogic_conf: NumalogicConf = MISSING


@dataclass
class DataStreamConf:
    name: str = "default"
    composite_keys: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    window_size: int = 12
    metric_configs: List[MetricConf] = field(default_factory=lambda: [MetricConf()])
    unified_config: UnifiedConf = field(default_factory=lambda: UnifiedConf())


@dataclass
class Configs:
    configs: List[DataStreamConf]


@dataclass
class PipelineConf:
    redis_conf: RedisConf
    registry_conf: RegistryConf
    prometheus_conf: PrometheusConf = MISSING
