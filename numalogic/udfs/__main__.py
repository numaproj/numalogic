import logging
import os
import sys

from omegaconf import OmegaConf, DictConfig
from pynumaflow.function import Server

from numalogic._constants import BASE_DIR
from numalogic.connectors.redis import get_redis_client_from_conf
from numalogic.udfs import PreprocessUDF, InferenceUDF, TrainerUDF, PostprocessUDF
from numalogic.udfs._config import PipelineConf


LOGGER = logging.getLogger(__name__)
CONFIG_DIR = os.path.join(BASE_DIR, "config")


def load_conf(filename: str) -> DictConfig:
    conf = OmegaConf.load(os.path.join(CONFIG_DIR, filename))
    schema = OmegaConf.structured(PipelineConf)
    conf = OmegaConf.merge(schema, conf)
    return OmegaConf.to_object(conf)


if __name__ == "__main__":
    step = sys.argv[1]

    conf_path = "config.yaml"
    try:
        conf_path = sys.argv[2]
    except (IndexError, TypeError):
        pass
    finally:
        LOGGER.info("Running %s with config path %s", step, conf_path)

    pipeline_conf = load_conf(conf_path)
    logging.info("Pipeline config: %s", pipeline_conf)
    redis_client = get_redis_client_from_conf(pipeline_conf.redis_conf)

    if step == "preprocess":
        udf = PreprocessUDF(
            r_client=redis_client, stream_conf=pipeline_conf.stream_confs["fciAsset"]
        )
    elif step == "inference":
        udf = InferenceUDF(r_client=redis_client, stream_confs=pipeline_conf.stream_confs)
    elif step == "trainer":
        udf = TrainerUDF(
            r_client=redis_client,
            druid_conf=pipeline_conf.druid_conf,
            stream_confs=pipeline_conf.stream_confs,
        )
    elif step == "postprocess":
        udf = PostprocessUDF(
            r_client=redis_client, stream_conf=pipeline_conf.stream_confs["fciAsset"]
        )
    else:
        raise ValueError(f"Invalid step: {step}")

    server = Server(map_handler=udf.exec)
    server.start()
