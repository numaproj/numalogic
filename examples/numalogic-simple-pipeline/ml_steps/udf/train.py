import logging
import os

import cachetools
import pandas as pd
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Datum, Messages, Message

from ml_steps.utility import Payload, save_model, TRAIN_DATA_PATH

LOGGER = logging.getLogger(__name__)
DEFAULT_WIN_SIZE = 12
WIN_SIZE = int(os.getenv("WIN_SIZE", DEFAULT_WIN_SIZE))
ttl_cache = cachetools.TTLCache(maxsize=128, ttl=20 * 60)


def train(key: str, datum: Datum):
    payload = Payload.from_json(datum.value.decode("utf-8"))
    messages = Messages()

    """ Checking if the model has already been triggerred for training. If so, we store 
        just the model key in TTL Cache. This is done to prevent triggering multiple 
        training for the same model as ML models might take time to train"""

    if "ae_model" in ttl_cache:
        messages.append(Message.to_drop())
        return messages
    ttl_cache["ae_model"] = "ae_model"

    # Load Training data
    data = pd.read_csv(TRAIN_DATA_PATH, index_col=None)

    # Preprocess training data
    preproc_transformer = LogTransformer()
    payload.data = preproc_transformer.fit_transform(data)

    # Train step
    pl = AutoencoderPipeline(model=VanillaAE(WIN_SIZE), seq_len=WIN_SIZE)
    pl.fit(data.to_numpy())
    LOGGER.info("%s - Training complete", payload.uuid)

    # Save to registry
    save_model(pl)
    LOGGER.info("%s - Model Saving complete", payload.uuid)

    # Train is the last vertex in the graph
    messages.append(Message.to_drop())
    return messages
