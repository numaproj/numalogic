import logging
import os

import cachetools
import numpy.typing as npt
import pandas as pd
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.udfs import NumalogicUDF
from numalogic.registry import MLflowRegistry
from numalogic.tools.data import TimeseriesDataModule
from numalogic.transforms import LogTransformer
from pynumaflow.function import Datum, Messages, Message

from src.utils import Payload, TRAIN_DATA_PATH
from typing import Optional

LOGGER = logging.getLogger(__name__)
WIN_SIZE = int(os.getenv("WIN_SIZE"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
TRACKING_URI = "http://mlflow-service.default.svc.cluster.local:5000"
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "50"))


class Trainer(NumalogicUDF):
    """UDF to train the model and save it in the registry."""

    ttl_cache = cachetools.TTLCache(maxsize=128, ttl=120 * 60)

    def __init__(self):
        super().__init__()
        self.registry = MLflowRegistry(tracking_uri=TRACKING_URI)
        self.model_key = "ae::model"

    def _save_artifact(
        self, model, skeys: list[str], dkeys: list[str], _: Optional[AutoencoderTrainer] = None
    ) -> None:
        """Saves the model in the registry."""
        self.registry.save(skeys=skeys, dkeys=dkeys, artifact=model)

    @staticmethod
    def _fit_preprocess(data: pd.DataFrame) -> npt.NDArray[float]:
        """Preprocesses the training data."""
        clf = LogTransformer()
        return clf.fit_transform(data.to_numpy())

    @staticmethod
    def _fit_threshold(data: npt.NDArray[float]) -> StdDevThreshold:
        """Fits the threshold model."""
        clf = StdDevThreshold()
        return clf.fit(data)

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        The train function here receives data from inference step.
        This step preprocesses, trains and saves a 1-D
        Convolution AE model into the registry.

        For more information about the arguments, refer:
        https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py

        Checking if the model has already been trained or is training in progress. If so, we store
        just the model key in TTL Cache. This is done to prevent multiple same model training
        triggers as ML models might take time to train.
        Reference: https://github.com/numaproj/numalogic/blob/main/numalogic/registry/mlflow_registry.py
        """
        payload = Payload.from_json(datum.value)

        # Check if request is already in local cache
        if self.model_key in self.ttl_cache:
            return Messages(Message.to_drop())
        self.ttl_cache[self.model_key] = self.model_key

        # Load Training data
        data = pd.read_csv(TRAIN_DATA_PATH, index_col=None)

        # Preprocess training data
        train_data = self._fit_preprocess(data)

        # Train the autoencoder model
        datamodule = TimeseriesDataModule(WIN_SIZE, train_data, batch_size=BATCH_SIZE)
        model = Conv1dAE(seq_len=WIN_SIZE, in_channels=train_data.shape[1])
        trainer = AutoencoderTrainer(max_epochs=MAX_EPOCHS, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        # Get reconstruction error of the training set
        train_reconerr = trainer.predict(model, dataloaders=datamodule.train_dataloader())

        LOGGER.info("%s - Training complete", payload.uuid)

        # Train the threshold model
        thresh_clf = self._fit_threshold(train_reconerr.numpy())

        # Save to registry
        self._save_artifact(model, ["ae"], ["model"], trainer)
        self._save_artifact(thresh_clf, ["thresh_clf"], ["model"])
        LOGGER.info("%s - Model Saving complete", payload.uuid)

        # Train is the last vertex in the graph
        return Messages(Message.to_drop())
