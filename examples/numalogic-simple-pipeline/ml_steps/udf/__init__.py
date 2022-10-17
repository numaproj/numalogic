import logging

from ml_steps.udf.inference import inference
from ml_steps.udf.input import input
from ml_steps.udf.postprocess import postprocess
from ml_steps.udf.preprocess import preprocess
from ml_steps.udf.train import train


__all__ = ["preprocess", "input", "inference", "postprocess", "train"]
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
