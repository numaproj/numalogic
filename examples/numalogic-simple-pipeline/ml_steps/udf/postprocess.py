import logging

import numpy as np
from numalogic.scores import tanh_norm
from pynumaflow.function import Messages, Message, Datum

from ml_steps.utility import Payload

LOGGER = logging.getLogger(__name__)


def postprocess(key: str, datum: Datum) -> Messages:
    # Load json data
    payload = Payload.from_json(datum.value.decode("utf-8"))

    # Postprocess step
    data = np.asarray(payload.ts_data)
    payload.anomaly_score = np.mean(tanh_norm(data))

    LOGGER.info("%s - PostProcess complete", payload.uuid)
    LOGGER.info("%s - The anomaly score is: %s", payload.uuid, payload.anomaly_score)

    # Convert Payload back to bytes
    messages = Messages()
    messages.append(Message.to_all(payload.to_json().encode("utf-8")))
    return messages
