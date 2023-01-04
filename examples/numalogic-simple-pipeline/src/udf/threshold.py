import logging

import numpy as np
from pynumaflow.function import Messages, Message, Datum

from src.utils import Payload, load_artifact

LOGGER = logging.getLogger(__name__)


def threshold(_: str, datum: Datum) -> Messages:
    r"""
    This UDF applies thresholding to the reconstruction error returned by the autoencoder.

    For more information about the arguments, refer:
    https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py
    """

    # Load data and convert bytes to Payload
    payload = Payload.from_json(datum.value.decode("utf-8"))
    messages = Messages()

    # Load the threshold model from registry
    thresh_clf_artifact = load_artifact(skeys=["thresh_clf"], dkeys=["model"])
    recon_err = np.asarray(payload.ts_data).reshape(-1, 1)

    # Check if model exists for inference
    if (not thresh_clf_artifact) or (not payload.is_artifact_valid):
        # If model not found, send it to trainer for training
        LOGGER.warning("%s - Model not found. Training the model.", payload.uuid)

        # Convert Payload back to bytes and conditional forward to train vertex
        payload.is_artifact_valid = False
        messages.append(Message.to_vtx(key="train", value=payload.to_json().encode("utf-8")))
        return messages

    LOGGER.debug("%s - Threshold Model found!", payload.uuid)

    thresh_clf = thresh_clf_artifact.artifact
    payload.ts_data = thresh_clf.predict(recon_err).tolist()

    LOGGER.info("%s - Thresholding complete", payload.uuid)

    # Convert Payload back to bytes and conditional forward to postprocess vertex
    messages.append(Message.to_vtx(key="postprocess", value=payload.to_json().encode("utf-8")))
    return messages
