import logging
import os

import cachetools
import pandas as pd
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Datum, Messages, Message

from src.utils import Payload, save_model, TRAIN_DATA_PATH

LOGGER = logging.getLogger(__name__)
WIN_SIZE = int(os.getenv("WIN_SIZE"))
ttl_cache = cachetools.TTLCache(maxsize=128, ttl=120 * 60)


def train(key: str, datum: Datum):
    payload = Payload.from_json(datum.value.decode("utf-8"))
    messages = Messages()

    """
    Checking if the model has already been trained or is training in progress. If so, we store 
    just the model key in TTL Cache. This is done to prevent multiple same model training triggers as 
    ML models might take time to train. In the key skey is 'ae' and dkey is 'model'. They are combined
    into a single key as 'ae::model'.
    """

    if "ae_model" in ttl_cache:
        messages.append(Message.to_drop())
        return messages
    ttl_cache["ae::model"] = "ae::model"

    # Load Training data
    data = pd.read_csv(TRAIN_DATA_PATH, index_col=None)

    # Preprocess training data
    clf = LogTransformer()
    train_data = clf.fit_transform(data)

    # Train step
    pl = AutoencoderPipeline(model=Conv1dAE(in_channels=1, enc_channels=12), seq_len=WIN_SIZE)
    pl.fit(train_data.to_numpy())
    LOGGER.info("%s - Training complete", payload.uuid)

    # Save to registry
    save_model(pl)
    LOGGER.info("%s - Model Saving complete", payload.uuid)

    # Train is the last vertex in the graph
    messages.append(Message.to_drop())
    return messages
