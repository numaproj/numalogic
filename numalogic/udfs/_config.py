import logging
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
from numalogic.tools.exceptions import ConfigNotFoundError

_logger = logging.getLogger(__name__)


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
    registry_conf: Optional[RegistryInfo] = field(
        default_factory=lambda: RegistryInfo(name="RedisRegistry", model_expiry_sec=172800)
    )
    prometheus_conf: Optional[PrometheusConf] = None
    druid_conf: Optional[DruidConf] = None


def load_pipeline_conf(*paths: str) -> PipelineConf:
    confs = []
    for _path in paths:
        try:
            conf = OmegaConf.load(_path)
        except FileNotFoundError:
            _logger.warning("Config file path: %s not found. Skipping...", _path)
            continue
        confs.append(conf)

    if not confs:
        _err_msg = f"None of the given conf paths exist: {paths}"
        raise ConfigNotFoundError(_err_msg)

    schema = OmegaConf.structured(PipelineConf)
    conf = OmegaConf.merge(schema, *confs)
    return OmegaConf.to_object(conf)
