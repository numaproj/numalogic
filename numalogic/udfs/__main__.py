import logging
import os
import sys
from typing import Final

from numalogic._constants import DEFAULT_BASE_CONF_PATH, DEFAULT_APP_CONF_PATH, DEFAULT_METRICS_PORT
from numalogic.connectors.redis import get_redis_client_from_conf
from numalogic.monitoring import start_metrics_server
from numalogic.udfs import load_pipeline_conf, UDFFactory, ServerFactory, set_logger

LOGGER = logging.getLogger(__name__)

BASE_CONF_FILE_PATH: Final[str] = os.getenv("BASE_CONF_PATH", default=DEFAULT_BASE_CONF_PATH)
APP_CONF_FILE_PATH: Final[str] = os.getenv("APP_CONF_PATH", default=DEFAULT_APP_CONF_PATH)
METRICS_PORT: Final[int] = int(os.getenv("METRICS_PORT", default=DEFAULT_METRICS_PORT))


def init_server(step: str, server_type: str):
    """Initializes and returns the server."""
    LOGGER.info("Merging config with file paths: %s, %s", BASE_CONF_FILE_PATH, APP_CONF_FILE_PATH)
    pipeline_conf = load_pipeline_conf(BASE_CONF_FILE_PATH, APP_CONF_FILE_PATH)
    LOGGER.info("Pipeline config: %s", pipeline_conf)

    LOGGER.info("Starting vertex with step: %s, server_type %s", step, server_type)
    if step == "mlpipeline":
        udf = UDFFactory.get_udf_instance(step, pl_conf=pipeline_conf)
    else:
        redis_client = get_redis_client_from_conf(pipeline_conf.redis_conf)
        udf = UDFFactory.get_udf_instance(step, r_client=redis_client, pl_conf=pipeline_conf)

    return ServerFactory.get_server_instance(server_type, handler=udf)


def start_server() -> None:
    """Starts the pynumaflow server."""
    set_logger()
    step = sys.argv[1]

    try:
        server_type = sys.argv[2]
    except (IndexError, TypeError):
        server_type = "sync"

    LOGGER.info("Running %s on %s server", step, server_type)

    # Start the metrics server at port METRICS_PORT = 8490
    start_metrics_server(METRICS_PORT)

    server = init_server(step, server_type)
    server.start()


if __name__ == "__main__":
    start_server()
