import os
from unittest.mock import patch

import fakeredis
from numalogic.config import NumalogicConf
from omegaconf import OmegaConf

from src._config import PipelineConf, Configs
from src._constants import TESTS_RESOURCES, CONFIG_DIR
from src.watcher import ConfigManager

server = fakeredis.FakeServer()
redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)


def mock_configs():
    schema: Configs = OmegaConf.structured(Configs)

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "users-config.yaml"))
    user_configs = OmegaConf.merge(schema, conf).configs

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "default-config.yaml"))
    default_configs = OmegaConf.merge(schema, conf).configs

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "numalogic_config.yaml"))
    schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
    default_numalogic = OmegaConf.merge(schema, conf)

    conf = OmegaConf.load(os.path.join(CONFIG_DIR, "default-configs", "pipeline_config.yaml"))
    schema: PipelineConf = OmegaConf.structured(PipelineConf)
    pipeline_config = OmegaConf.merge(schema, conf)

    return user_configs, default_configs, default_numalogic, pipeline_config


with patch("src.connectors.sentinel.get_redis_client") as mock_get_redis_client:
    mock_get_redis_client.return_value = redis_client
    with patch(
        "src.connectors.sentinel.get_redis_client_from_conf"
    ) as mock_get_redis_client_from_conf:
        mock_get_redis_client_from_conf.return_value = redis_client
        with patch.object(ConfigManager, "load_configs") as mock_confs:
            mock_confs.return_value = mock_configs()
            from src.udf import Preprocess, Inference, Threshold, Postprocess
            from src.udsink import Train


__all__ = [
    "redis_client",
    "Train",
    "Preprocess",
    "Inference",
    "Threshold",
    "Postprocess",
    "mock_configs",
]
