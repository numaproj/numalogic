import logging
import os

from numalogic.udfs._base import NumalogicUDF
from numalogic.udfs.inference import InferenceUDF
from numalogic.udfs.trainer import TrainerUDF
from numalogic.udfs.preprocess import PreprocessUDF
from numalogic.udfs.postprocess import PostprocessUDF
from numalogic.udfs.factory import UDFFactory, ServerFactory
from numalogic.udfs._config import StreamConf, PipelineConf, load_pipeline_conf


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
]


if os.getenv("DEBUG"):
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
