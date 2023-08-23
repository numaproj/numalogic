from dataclasses import dataclass, field

from numalogic.config import NumalogicConf
from numalogic.connectors._config import ConnectorConf, ConnectorType


@dataclass
class StreamConf:
    config_id: str = "default"
    window_size: int = 12
    composite_keys: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    numalogic_conf: NumalogicConf = field(default_factory=lambda: NumalogicConf())


@dataclass
class PipelineConf:
    stream_confs: dict[str, StreamConf] = field(default_factory=dict)
    connector_confs: dict[ConnectorType, ConnectorConf] = field(default_factory=dict)
