import os
import sys

from omegaconf import OmegaConf, DictConfig
from pynumaflow.function import Server

from numalogic._constants import BASE_DIR
from numalogic.connectors.redis import get_redis_client_from_conf
from numalogic.udfs import PreprocessUDF, InferenceUDF, TrainerUDF, PostprocessUDF
from numalogic.udfs._config import PipelineConf


CONFIG_DIR = os.path.join(BASE_DIR, "config")


def load_conf(filename: str = "conf.yaml") -> DictConfig:
    conf = OmegaConf.load(os.path.join(CONFIG_DIR, filename))
    schema = OmegaConf.structured(PipelineConf)
    return OmegaConf.merge(schema, conf)


def get_redis_client():
    global redis_client

    if redis_client:
        return redis_client


if __name__ == "__main__":
    step = sys.argv[1]

    pipeline_conf = load_conf()
    redis_client = get_redis_client_from_conf(pipeline_conf.redis_conf)

    if step == "preprocess":
        udf = PreprocessUDF(r_client=redis_client, stream_conf=pipeline_conf.stream_confs[0])
    elif step == "inference":
        udf = InferenceUDF(r_client=redis_client, stream_confs=pipeline_conf.stream_confs)
    elif step == "trainer":
        udf = TrainerUDF(
            r_client=redis_client,
            druid_conf=pipeline_conf.druid_conf,
            stream_confs=pipeline_conf.stream_confs,
        )
    elif step == "postprocess":
        udf = PostprocessUDF(r_client=redis_client, stream_conf=pipeline_conf.stream_confs[0])
    else:
        raise ValueError(f"Invalid step: {step}")

    server = Server(map_handler=udf.exec)
    server.start()
