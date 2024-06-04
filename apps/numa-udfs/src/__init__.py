import logging
from logging import config as logconf
import os

from src.constants import SRC_DIR, LOG_CONF_PATH
from src._base import NumalogicUDF
from src._config import StreamConf, PipelineConf, MLPipelineConf, load_pipeline_conf
from src.factory import UDFFactory, ServerFactory
from src.payloadtx import PayloadTransformer
from src.inference import InferenceUDF
from src.postprocess import PostprocessUDF
from src.preprocess import PreprocessUDF
from src.trainer import TrainerUDF, PromTrainerUDF, DruidTrainerUDF, RDSTrainerUDF


def set_logger() -> None:
    """Sets the logger for the UDFs."""
    logconf.fileConfig(
        fname=LOG_CONF_PATH,
        disable_existing_loggers=False,
    )
    if os.getenv("DEBUG", "false").lower() == "true":
        logging.getLogger("root").setLevel(logging.DEBUG)


__all__ = [
    "NumalogicUDF",
    "PayloadTransformer",
    "PreprocessUDF",
    "InferenceUDF",
    "TrainerUDF",
    "PromTrainerUDF",
    "DruidTrainerUDF",
    "RDSTrainerUDF",
    "PostprocessUDF",
    "UDFFactory",
    "StreamConf",
    "PipelineConf",
    "MLPipelineConf",
    "load_pipeline_conf",
    "ServerFactory",
    "set_logger",
]
