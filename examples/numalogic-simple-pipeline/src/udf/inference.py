import logging
import os

import numpy as np
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.autoencoder.variants import Conv1dAE
from pynumaflow.function import Messages, Message, Datum

from src.utils import Payload, load_artifact

LOGGER = logging.getLogger(__name__)
WIN_SIZE = int(os.getenv("WIN_SIZE"))


def inference(key: str, datum: Datum) -> Messages:
    r"""
    Here inference is done on the data, given, the ML model is present
    in the registry. If a model does not exist, conditional forward the payload to
    train vertex for ML model training. Otherwise, conditional forward the inferred data
    to postprocess vertex for generating anomaly score for the payload.

    For more information about the arguments, refer:
    https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py
    """

    # Load data and convert bytes to Payload
    payload = Payload.from_json(datum.value.decode("utf-8"))
    messages = Messages()

    artifact_data = load_artifact(skeys=["ae"], dkeys=["model"], type="pytorch")
    thresh_clf_data = load_artifact(skeys=["thresh_clf"], dkeys=["model"])

    # Check if model exists for inference
    if artifact_data and thresh_clf_data:
        LOGGER.info("%s - Model found!", payload.uuid)

        # Load model from registry
        pl = AutoencoderPipeline(model=Conv1dAE(in_channels=1, enc_channels=12), seq_len=WIN_SIZE)
        pl.load(model=artifact_data.artifact)

        # Load the threshold model from registry
        thresh_clf = thresh_clf_data.artifact

        # Infer using the loaded model
        infer_data = np.asarray(payload.ts_data).reshape(-1, 1)
        score_data = pl.score(infer_data)
        payload.ts_data = thresh_clf.predict(score_data).tolist()

        LOGGER.info("%s - Inference complete", payload.uuid)

        # Convert Payload back to bytes and conditional forward to postprocess vertex
        messages.append(Message.to_vtx(key="postprocess", value=payload.to_json().encode("utf-8")))

    # If model not found, send it to trainer for training
    else:
        LOGGER.exception("%s - Model not found. Training the model.", payload.uuid)

        # Convert Payload back to bytes and conditional forward to train vertex
        messages.append(Message.to_vtx(key="train", value=payload.to_json().encode("utf-8")))

    return messages
