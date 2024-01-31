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
from numalogic.udfs.entities import ScoreFunc

_logger = logging.getLogger(__name__)


@dataclass
class UnifiedScoreConf:
    """
    A data class representing the configuration for unified scoring.

    Args:
        scoring_function (str): The scoring function to be used. Defaults to "max".
    """

    scoring_function: ScoreFunc = ScoreFunc.MAX


@dataclass
class MLPipelineConf:
    """
    A data class representing the configuration for an ML pipeline.

    Args:
        pipeline_id (str): The ID of the pipeline. Defaults to "default".
        metrics (List[str]): A list of metrics to be used in the pipeline.
        unified_scoring_conf (UnifiedScoreConf): The configuration for unified scoring.
        numalogic_conf (NumalogicConf): The configuration for Numalogic.
    """

    pipeline_id: str = "default"
    metrics: list[str] = field(default_factory=list)
    unified_scoring_conf: UnifiedScoreConf = field(default_factory=lambda: UnifiedScoreConf())
    numalogic_conf: NumalogicConf = field(default_factory=lambda: NumalogicConf())


@dataclass
class StreamConf:
    """
    A data class representing the configuration for a stream.

    Args:
        config_id (str): The ID of the stream configuration.
        source (ConnectorType): The type of data source connector to be used.
        window_size (int): The window size for stream processing. Defaults to 12.
        composite_keys (List[str]): A list of composite keys for stream processing.
        ml_pipelines (Dict[str, MLPipelineConf]): A dictionary of ML pipeline configurations
    """

    config_id: str = "default"
    source: ConnectorType = ConnectorType.druid  # TODO: do not allow redis connector here
    window_size: int = 12
    composite_keys: list[str] = field(default_factory=list)
    ml_pipelines: dict[str, MLPipelineConf] = field(default_factory=dict)


@dataclass
class PipelineConf:
    """
    A data class representing the configuration for a pipeline.

    Args:

        stream_confs (Dict[str, StreamConf]): Dictionary of stream_confs associated with pipeline.
        redis_conf (Optional[RedisConf]): The configuration for Redis.
        registry_conf (Optional[RegistryInfo]): The configuration for the registry.
        prometheus_conf (Optional[PrometheusConf]): The configuration for Prometheus.
        druid_conf (Optional[DruidConf]): The configuration for Druid.
    """

    stream_confs: dict[str, StreamConf] = field(default_factory=dict)
    redis_conf: Optional[RedisConf] = None
    registry_conf: Optional[RegistryInfo] = field(
        default_factory=lambda: RegistryInfo(name="RedisRegistry", model_expiry_sec=172800)
    )
    prometheus_conf: Optional[PrometheusConf] = None
    druid_conf: Optional[DruidConf] = None


def load_pipeline_conf(*paths: str) -> PipelineConf:
    """
    Load the pipeline configuration from the specified paths and return a `PipelineConf` object.

    Args:
        *paths (str): Variable length argument representing the paths to the configuration files.

    Returns
    -------
        PipelineConf: The loaded pipeline configuration.

    Raises
    ------
        ConfigNotFoundError: If none of the given configuration paths exist.
    """
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
