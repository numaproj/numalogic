from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf

from numalogic.config import NumalogicConf, RegistryInfo

from numalogic.connectors import (
    ConnectorType,
    RedisConf,
    PrometheusConf,
    DruidConf,
)


@dataclass
class MLPipelineConf:
    pipeline_id: str = "default"
    metrics: list[str] = field(default_factory=list)
    numalogic_conf: NumalogicConf = field(default_factory=lambda: NumalogicConf())


@dataclass
class StreamConf:
    config_id: str = "default"
    source: ConnectorType = ConnectorType.druid  # TODO: do not allow redis connector here
    window_size: int = 12
    composite_keys: list[str] = field(default_factory=list)
    ml_pipelines: dict[str, MLPipelineConf] = field(default_factory=dict)


@dataclass
class StreamPipelineConf:
    stream_confs: dict[str, StreamConf] = field(default_factory=dict)
    redis_conf: Optional[RedisConf] = None
    registry_conf: Optional[RegistryInfo] = field(
        default_factory=lambda: RegistryInfo(name="RedisRegistry", model_expiry_sec=172800)
    )
    prometheus_conf: Optional[PrometheusConf] = None
    druid_conf: Optional[DruidConf] = None


def load_pipeline_conf(path: str) -> StreamPipelineConf:
    conf = OmegaConf.load(path)
    schema = OmegaConf.structured(StreamPipelineConf)
    conf = OmegaConf.merge(schema, conf)
    return OmegaConf.to_object(conf)
