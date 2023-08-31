from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf

from numalogic.config import NumalogicConf
from numalogic.connectors import (
    ConnectorType,
    RedisConf,
    PrometheusConf,
    DruidConf,
)


@dataclass
class StreamConf:
    config_id: str = "default"
    source: ConnectorType = ConnectorType.druid  # TODO: do not allow redis connector here
    window_size: int = 12
    composite_keys: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    numalogic_conf: NumalogicConf = field(default_factory=lambda: NumalogicConf())


@dataclass
class PipelineConf:
    stream_confs: dict[str, StreamConf] = field(default_factory=dict)
    redis_conf: Optional[RedisConf] = None
    prometheus_conf: Optional[PrometheusConf] = None
    druid_conf: Optional[DruidConf] = None


def load_pipeline_conf(path: str) -> PipelineConf:
    conf = OmegaConf.load(path)
    schema = OmegaConf.structured(PipelineConf)
    conf = OmegaConf.merge(schema, conf)
    return OmegaConf.to_object(conf)
