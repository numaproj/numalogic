import logging
from logging import config as logconf
import os

from numalogic._constants import BASE_DIR
from numalogic.udfs._base import NumalogicUDF
from numalogic.udfs._config import StreamConf, PipelineConf, load_pipeline_conf
from numalogic.udfs.factory import UDFFactory, ServerFactory
from numalogic.udfs.inference import InferenceUDF
from numalogic.udfs.postprocess import PostprocessUDF
from numalogic.udfs.preprocess import PreprocessUDF
from numalogic.udfs.trainer import TrainerUDF


def set_logger() -> None:
    """Sets the logger for the UDFs."""
    logconf.fileConfig(
        fname=os.path.join(BASE_DIR, "log.conf"),
        disable_existing_loggers=False,
    )
    if os.getenv("DEBUG", "false").lower() == "true":
        logging.getLogger("root").setLevel(logging.DEBUG)


__all__ = [
    "NumalogicUDF",
    "PreprocessUDF",
    "InferenceUDF",
    "TrainerUDF",
    "PostprocessUDF",
    "UDFFactory",
    "StreamConf",
    "PipelineConf",
    "load_pipeline_conf",
    "ServerFactory",
    "set_logger",
]
