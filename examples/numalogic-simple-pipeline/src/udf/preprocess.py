import json
import logging
import uuid

import numpy as np
from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Messages, Message, Datum

from src.utility import Payload

LOGGER = logging.getLogger(__name__)


def preprocess(key: str, datum: Datum) -> Messages:
    # Load json data
    json_data = datum.value
    ts_array = json.loads(json_data)["data"]
    payload = Payload(ts_data=ts_array, uuid=str(uuid.uuid4()))

    # preprocess step
    data = np.asarray(payload.ts_data)
    clf = LogTransformer()
    payload.ts_data = clf.transform(data).tolist()
    LOGGER.info("%s - Preprocess complete", payload.uuid)

    # Convert Payload back to bytes
    messages = Messages()
    messages.append(Message.to_all(payload.to_json().encode("utf-8")))
    return messages
