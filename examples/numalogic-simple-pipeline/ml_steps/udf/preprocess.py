import logging

import numpy as np
from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Messages, Message, Datum

from ml_steps.dtypes import Status, Payload

LOGGER = logging.getLogger(__name__)

PRE_PROC = LogTransformer()


def preprocess(key: str, datum: Datum) -> Messages:

    # Load json data
    payload = Payload.from_json(datum.value.decode("utf-8"))

    # preprocess step
    data = np.asarray(payload.data)
    preproc_transformer = LogTransformer()
    payload.data = preproc_transformer.transform(data).tolist()
    payload.status = Status.PRE_PROCESSED.value
    LOGGER.info("%s - Preprocess complete", payload.uuid)

    # Convert Payload back to bytes
    messages = Messages()
    messages.append(Message.to_all(payload.to_json().encode("utf-8")))
    return messages
