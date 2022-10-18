import logging
import os

import numpy as np
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.autoencoder.variants import VanillaAE
from pynumaflow.function import Messages, Message, Datum

from ml_steps.dtypes import Status, Payload, MODEL_PATH

LOGGER = logging.getLogger(__name__)
DEFAULT_WIN_SIZE = 12
WIN_SIZE = int(os.getenv("WIN_SIZE", DEFAULT_WIN_SIZE))


def inference(key: str, datum: Datum) -> Messages:
    # Load data and convert bytes to MessagePacket
    payload = Payload.from_json(datum.value.decode("utf-8"))

    # Load the model for the given metric name
    messages = Messages()

    # Check if model exists in the path
    model_path = MODEL_PATH
    try:
        pl = AutoencoderPipeline(model=VanillaAE(WIN_SIZE), seq_len=WIN_SIZE)
        pl.load(path=model_path)
        LOGGER.info("%s - Model found!", payload.uuid)
        payload.data = pl.score(payload.data).tolist()
        payload.status = Status.INFERRED.value
        LOGGER.info("%s - Inference complete", payload.uuid)

        # Convert Payload back to bytes and send it to postprocess vertex
        messages.append(
            Message.to_vtx(key="postprocess", value=payload.to_json().encode("utf-8"))
        )
    # If model is not present send it to trainer for training
    except Exception as ex:
        LOGGER.exception(
            "%s - Model not found. Retraining the model. Error:%r",
            payload.uuid,
            ex,
        )
        payload.status = Status.TRAIN.value
        # Convert Payload back to bytes and send it to train vertex
        messages.append(
            Message.to_vtx(key="train", value=payload.to_json().encode("utf-8"))
        )

    return messages
