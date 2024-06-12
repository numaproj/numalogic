import logging
from logging import config as logconf
import os

from numalogic._constants import BASE_DIR
from numalogic.udfs._base import NumalogicUDF
from numalogic.udfs._config import StreamConf, PipelineConf, MLPipelineConf, load_pipeline_conf
from numalogic.udfs._metrics_utility import MetricsLoader
from numalogic.udfs.factory import UDFFactory, ServerFactory
from numalogic.udfs.payloadtx import PayloadTransformer
from numalogic.udfs.inference import InferenceUDF
from numalogic.udfs.postprocess import PostprocessUDF
from numalogic.udfs.preprocess import PreprocessUDF
from numalogic.udfs.trainer import TrainerUDF, PromTrainerUDF, DruidTrainerUDF, RDSTrainerUDF


def set_logger() -> None:
    """Sets the logger for the UDFs."""
    logconf.fileConfig(
        fname=os.path.join(BASE_DIR, "log.conf"),
        disable_existing_loggers=False,
    )
    if os.getenv("DEBUG", "false").lower() == "true":
        logging.getLogger("root").setLevel(logging.DEBUG)


def set_metrics(conf_file: str) -> None:
    """Sets the metrics for the UDFs."""
    MetricsLoader().load_metrics(config_file_path=conf_file)


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
    "set_metrics",
    "MetricsLoader",
]
