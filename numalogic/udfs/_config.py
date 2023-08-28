from dataclasses import dataclass, field
from typing import Optional

from numalogic.config import NumalogicConf
from numalogic.connectors._config import (
    ConnectorType,
    RedisConf,
    PrometheusConf,
    DruidConf,
)


@dataclass
class StreamConf:
    config_id: str = "default"
    source: ConnectorType = ConnectorType.druid
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
