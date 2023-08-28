import logging
import os
import sys

from pynumaflow.function import Server

from numalogic._constants import BASE_CONF_DIR
from numalogic.connectors.redis import get_redis_client_from_conf
from numalogic.udfs import (
    PreprocessUDF,
    InferenceUDF,
    TrainerUDF,
    PostprocessUDF,
    load_pipeline_conf,
)

LOGGER = logging.getLogger(__name__)
CONF_FILE_PATH = os.getenv(
    "CONF_PATH", default=os.path.join(BASE_CONF_DIR, "default-configs", "config.yaml")
)


if __name__ == "__main__":
    step = sys.argv[1]

    LOGGER.info("Running %s with config path %s", step, CONF_FILE_PATH)

    pipeline_conf = load_pipeline_conf(CONF_FILE_PATH)
    logging.info("Pipeline config: %s", pipeline_conf)

    redis_client = get_redis_client_from_conf(pipeline_conf.redis_conf)

    if step == "preprocess":
        udf = PreprocessUDF(r_client=redis_client, stream_confs=pipeline_conf.stream_confs)
    elif step == "inference":
        udf = InferenceUDF(r_client=redis_client, stream_confs=pipeline_conf.stream_confs)
    elif step == "trainer":
        udf = TrainerUDF(
            r_client=redis_client,
            druid_conf=pipeline_conf.druid_conf,
            stream_confs=pipeline_conf.stream_confs,
        )
    elif step == "postprocess":
        udf = PostprocessUDF(r_client=redis_client, stream_confs=pipeline_conf.stream_confs)
    else:
        raise ValueError(f"Invalid step: {step}")

    server = Server(map_handler=udf.exec)
    server.start()
