import logging
from logging import config as logconf
import os


from numalogic._constants import BASE_DIR
from numalogic.udfs._base import NumalogicUDF
from numalogic.udfs._config import StreamConf, PipelineConf, MLPipelineConf, load_pipeline_conf
from numalogic.udfs.factory import UDFFactory, ServerFactory
from numalogic.udfs.payloadtx import PayloadTransformer
from numalogic.udfs.inference import InferenceUDF
from numalogic.udfs.postprocess import PostprocessUDF
from numalogic.udfs.preprocess import PreprocessUDF
from numalogic.udfs.trainer import TrainerUDF, PromTrainerUDF, DruidTrainerUDF


class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console logs."""

    WHITE = "\x1b[1m"
    BLUE = "\x1b[1;36m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._formats = {
            logging.DEBUG: f"{self.BLUE}{self._fmt}{self.RESET}",
            logging.INFO: f"{self.WHITE}{self._fmt}{self.RESET}",
            logging.WARNING: f"{self.YELLOW}{self._fmt}{self.RESET}",
            logging.ERROR: f"{self.RED}{self._fmt}{self.RESET}",
            logging.CRITICAL: f"{self.BOLD_RED}{self._fmt}{self.RESET}",
        }

    def format(self, record):
        self._style._fmt = self._formats.get(record.levelno)
        return super().format(record)


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
    "PayloadTransformer",
    "PreprocessUDF",
    "InferenceUDF",
    "TrainerUDF",
    "PromTrainerUDF",
    "DruidTrainerUDF",
    "PostprocessUDF",
    "UDFFactory",
    "StreamConf",
    "PipelineConf",
    "MLPipelineConf",
    "load_pipeline_conf",
    "ServerFactory",
    "set_logger",
]
