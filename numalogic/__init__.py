import logging


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

LOGGER.addHandler(stream_handler)
