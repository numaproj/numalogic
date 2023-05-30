import logging
import os

import cachetools
import pandas as pd
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.preprocess.transformer import LogTransformer
from numalogic.tools.data import TimeseriesDataModule
from pynumaflow.function import Datum, Messages, Message

from src.utils import Payload, save_artifact, TRAIN_DATA_PATH

LOGGER = logging.getLogger(__name__)
WIN_SIZE = int(os.getenv("WIN_SIZE"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
ttl_cache = cachetools.TTLCache(maxsize=128, ttl=120 * 60)


def train(key: str, datum: Datum):
    r"""The train function here receives data from inference step.
    This step preprocesses, trains and saves a 1-D
    Convolution AE model into the registry.

    For more information about the arguments, refer:
    https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py
    """
    payload = Payload.from_json(datum.value.decode("utf-8"))
    messages = Messages()

    """
    Checking if the model has already been trained or is training in progress. If so, we store
    just the model key in TTL Cache. This is done to prevent multiple same model training
    triggers as ML models might take time to train. In the key skey = ['ae'] and dkey = ['model'].
    They are combined into a single key as 'ae::model'.
    Reference: https://github.com/numaproj/numalogic/blob/main/numalogic/registry/mlflow_registry.py
    """

    model_key = "ae::model"

    if model_key in ttl_cache:
        messages.append(Message.to_drop())
        return messages

    ttl_cache[model_key] = model_key

    # Load Training data
    data = pd.read_csv(TRAIN_DATA_PATH, index_col=None)

    # Preprocess training data
    clf = LogTransformer()
    train_data = clf.fit_transform(data.to_numpy())

    # Train step
    datamodule = TimeseriesDataModule(WIN_SIZE, train_data, batch_size=BATCH_SIZE)
    model = Conv1dAE(seq_len=WIN_SIZE, in_channels=train_data.shape[1], enc_channels=12)
    trainer = AutoencoderTrainer(max_epochs=50, enable_progress_bar=True)
    trainer.fit(model, datamodule=datamodule)

    train_reconerr = trainer.predict(model, dataloaders=datamodule.train_dataloader())

    LOGGER.info("%s - Training complete", payload.uuid)

    # Define Threshold method
    thresh_clf = StdDevThreshold(std_factor=1.2)
    thresh_clf.fit(train_reconerr.numpy())

    # Save to registry
    save_artifact(model, skeys=["ae"], dkeys=["model"])
    save_artifact(thresh_clf, skeys=["thresh_clf"], dkeys=["model"])
    LOGGER.info("%s - Model Saving complete", payload.uuid)

    # Train is the last vertex in the graph
    messages.append(Message.to_drop())
    return messages
