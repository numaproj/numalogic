import json
import logging
import uuid

import numpy as np
from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Messages, Message, Datum

from src.utils import Payload

LOGGER = logging.getLogger(__name__)


def preprocess(_: str, datum: Datum) -> Messages:
    r"""
    The preprocess function here transforms the input data for ML inference and sends
    the payload to inference vertex.

    For more information about the arguments, refer:
    https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py
    """

    # Load json data
    json_data = datum.value
    ts_array = json.loads(json_data)["data"]
    payload = Payload(ts_data=ts_array, uuid=str(uuid.uuid4()))

    # preprocess step
    data = np.asarray(payload.ts_data)
    clf = LogTransformer()
    payload.ts_data = clf.transform(data).tolist()
    LOGGER.info("%s - Preprocess complete for data: %s", payload.uuid, payload.ts_data)

    # Convert Payload back to bytes
    return Messages(Message.to_all(payload.to_json().encode("utf-8")))
