import logging
from ml_steps.pl_factory import ModelPlFactory
from ml_steps.aepipeline import AutoencoderPipeline, SparseAEPipeline

__all__ = [
    "AutoencoderPipeline",
    "SparseAEPipeline",
    "ModelPlFactory",
]
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

LOGGER.addHandler(stream_handler)
