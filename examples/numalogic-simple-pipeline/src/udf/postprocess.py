import logging

import numpy as np
from numalogic.scores import tanh_norm
from pynumaflow.function import Messages, Message, Datum

from src.utility import Payload

LOGGER = logging.getLogger(__name__)


def postprocess(key: str, datum: Datum) -> Messages:
    # Load json data
    payload = Payload.from_json(datum.value.decode("utf-8"))

    # Postprocess step
    data = np.asarray(payload.ts_data)

    # Taking mean of the anomaly scores
    payload.anomaly_score = tanh_norm(np.mean(data))

    LOGGER.info("%s - The anomaly score is: %s", payload.uuid, payload.anomaly_score)

    # Convert Payload back to bytes
    messages = Messages()
    messages.append(Message.to_all(payload.to_json().encode("utf-8")))
    return messages
