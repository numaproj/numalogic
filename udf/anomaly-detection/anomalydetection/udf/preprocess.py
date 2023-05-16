import json
from typing import List

from pynumaflow.function import Messages, Message, Datum

from anomalydetection import get_logger

_LOGGER = get_logger(__name__)


def preprocess(keys: List[str], datum: Datum) -> Messages:
    _ = datum.event_time
    _ = datum.watermark
    messages = Messages()
    _LOGGER.debug("Received Msg: { keys: %s, value: %s }", datum.value, keys)

    return messages
