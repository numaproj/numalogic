import uuid

import numpy as np
from pynumaflow.function import Messages, Datum, Message

from ml_steps.dtypes import Status, Payload

DEFAULT_WIN_SIZE = 12


def input(key: str, datum: Datum) -> Messages:
    messages = Messages()

    payload = Payload(
        data=np.random.rand(12, 1).tolist(),
        status=Status.RAW.value,
        uuid=str(uuid.uuid4()),
    )

    messages.append(Message.to_all(payload.to_json().encode("utf-8")))
    return messages
