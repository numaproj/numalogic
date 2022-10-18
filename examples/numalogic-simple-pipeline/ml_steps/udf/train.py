import logging
import os

import pandas as pd
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Datum, Messages, Message

from ml_steps.dtypes import Payload, MODEL_PATH, TRAIN_DATA_PATH

LOGGER = logging.getLogger(__name__)
DEFAULT_WIN_SIZE = 12
WIN_SIZE = int(os.getenv("WIN_SIZE", DEFAULT_WIN_SIZE))


def train(key: str, datum: Datum):
    payload = Payload.from_json(datum.value.decode("utf-8"))

    # Load Training data
    data = pd.read_csv(TRAIN_DATA_PATH, index_col=None)

    # Preprocess training data
    preproc_transformer = LogTransformer()
    payload.data = preproc_transformer.fit_transform(data)

    # Train step
    pl = AutoencoderPipeline(model=VanillaAE(WIN_SIZE), seq_len=WIN_SIZE)
    pl.fit(data.reshape(-1, 1))
    LOGGER.info("%s - Training complete", payload.uuid)
    pl.save(path=MODEL_PATH)
    LOGGER.info("%s - Model Saving complete", payload.uuid)

    # Convert Payload back to bytes
    messages = Messages()
    messages.append(Message.to_all(payload.to_json().encode("utf-8")))
    return messages
