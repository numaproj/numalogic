import logging
import os

from src._config import UnifiedConf, StreamConf, PipelineConf, Configs


def get_logger(name):
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    if os.getenv("DEBUG", False):
        logger.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)

    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    pl_logger = logging.getLogger("pytorch_lightning")
    pl_logger.propagate = False
    pl_logger.setLevel(logging.ERROR)
    pl_logger.addHandler(stream_handler)
    return logger


__all__ = ["get_logger", "UnifiedConf", "StreamConf", "Configs", "PipelineConf"]
