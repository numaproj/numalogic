import logging
import os
import sys

from numalogic._constants import BASE_CONF_DIR
from numalogic.connectors.redis import get_redis_client_from_conf
from numalogic.udfs import load_pipeline_conf, UDFFactory, ServerFactory, set_logger

LOGGER = logging.getLogger(__name__)

# TODO support user config paths
CONF_FILE_PATH = os.getenv(
    "CONF_PATH", default=os.path.join(BASE_CONF_DIR, "default-configs", "config.yaml")
)


def start_server():
    """Starts the pynumaflow server."""
    set_logger()
    step = sys.argv[1]

    try:
        server_type = sys.argv[2]
    except (IndexError, TypeError):
        server_type = "sync"

    LOGGER.info("Running %s on %s server with config path %s", step, server_type, CONF_FILE_PATH)

    pipeline_conf = load_pipeline_conf(CONF_FILE_PATH)
    logging.info("Pipeline config: %s", pipeline_conf)

    redis_client = get_redis_client_from_conf(pipeline_conf.redis_conf)
    udf = UDFFactory.get_udf_instance(step, r_client=redis_client, pl_conf=pipeline_conf)

    server = ServerFactory.get_server_instance(server_type, map_handler=udf)
    server.start()


if __name__ == "__main__":
    start_server()
