import logging
import os

import numpy as np
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.autoencoder.variants import VanillaAE
from pynumaflow.function import Messages, Message, Datum

from ml_steps.utility import Payload, load_model

LOGGER = logging.getLogger(__name__)
DEFAULT_WIN_SIZE = 12
WIN_SIZE = int(os.getenv("WIN_SIZE", DEFAULT_WIN_SIZE))


def inference(key: str, datum: Datum) -> Messages:

    # Load data and convert bytes to MessagePacket
    payload = Payload.from_json(datum.value.decode("utf-8"))
    messages = Messages()

    artifact = load_model()

    # Check if model exists for inference
    if artifact:
        # load model from registry
        pl = AutoencoderPipeline(model=VanillaAE(WIN_SIZE), seq_len=WIN_SIZE)
        pl.load(model=artifact["primary_artifact"], **artifact["metadata"])

        LOGGER.info("%s - Model found!", payload.uuid)

        # Infer using the loaded model
        infer_data = np.asarray(payload.ts_data).reshape(-1, 1)
        payload.ts_data = pl.score(infer_data).tolist()

        LOGGER.info("%s - Inference complete", payload.uuid)

        # Convert Payload back to bytes and send it to postprocess vertex
        messages.append(Message.to_vtx(key="postprocess", value=payload.to_json().encode("utf-8")))

    # If model not found, send it to trainer for training
    else:
        LOGGER.exception("%s - Model not found. Training the model.", payload.uuid)

        # Convert Payload back to bytes and send it to train vertex
        messages.append(Message.to_vtx(key="train", value=payload.to_json().encode("utf-8")))

    return messages
