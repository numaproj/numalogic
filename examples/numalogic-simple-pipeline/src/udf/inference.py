import logging
import os

import numpy as np
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.tools.data import StreamingDataset
from pynumaflow.function import Messages, Message, Datum
from torch.utils.data import DataLoader

from src.utils import Payload, load_artifact

LOGGER = logging.getLogger(__name__)
WIN_SIZE = int(os.getenv("WIN_SIZE"))


def inference(_: str, datum: Datum) -> Messages:
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

    artifact_data = load_artifact(skeys=["ae"], dkeys=["model"], type_="pytorch")
    thresh_clf_data = load_artifact(skeys=["thresh_clf"], dkeys=["model"])

    stream_data = np.asarray(payload.ts_data).reshape(-1, 1)

    # Check if model exists for inference
    if artifact_data and thresh_clf_data:
        LOGGER.info("%s - Model found!", payload.uuid)

        # Load model from registry
        main_model = artifact_data.artifact
        streamloader = DataLoader(StreamingDataset(stream_data, WIN_SIZE))

        trainer = AutoencoderTrainer()
        recon_err = trainer.predict(main_model, dataloaders=streamloader)

        # Load the threshold model from registry
        thresh_clf = thresh_clf_data.artifact
        payload.ts_data = thresh_clf.predict(recon_err).tolist()

        LOGGER.info("%s - Inference complete", payload.uuid)

        # Convert Payload back to bytes and conditional forward to postprocess vertex
        messages.append(Message.to_vtx(key="postprocess", value=payload.to_json().encode("utf-8")))
        return messages

    # If model not found, send it to trainer for training
    LOGGER.warning("%s - Model not found. Training the model.", payload.uuid)

    # Convert Payload back to bytes and conditional forward to train vertex
    messages.append(Message.to_vtx(key="train", value=payload.to_json().encode("utf-8")))
    return messages
