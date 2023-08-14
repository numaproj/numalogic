import logging

import pandas as pd
from cachetools import TTLCache
from numalogic.blocks import (
    BlockPipeline,
    PreprocessBlock,
    NNBlock,
    ThresholdBlock,
    PostprocessBlock,
)
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.udfs import NumalogicUDF
from numalogic.registry import RedisRegistry
from numalogic.transforms import TanhNorm
from pynumaflow.function import Datum, Messages, Message
from sklearn.preprocessing import StandardScaler

from src.utils import RedisClient, TRAIN_DATA_PATH

_LOGGER = logging.getLogger(__name__)


class Train(NumalogicUDF):
    """UDF to train the model and save it to the registry."""

    ttl_cache = TTLCache(maxsize=16, ttl=60)

    def __init__(self, seq_len: int = 12, num_series: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = num_series
        self.registry = RedisRegistry(client=RedisClient().get_client())
        self._model_key = "sparsevanillae"

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """The train function here trains the model and saves it to the registry."""
        # Check if a training message has been received very recently
        if self._model_key in self.ttl_cache:
            return Messages(Message.to_drop())
        self.ttl_cache[self._model_key] = self._model_key

        # Load Training data
        data = pd.read_csv(TRAIN_DATA_PATH, index_col=None)

        # Define the block pipeline
        block_pl = BlockPipeline(
            PreprocessBlock(StandardScaler()),
            NNBlock(
                SparseVanillaAE(seq_len=self.seq_len, n_features=self.n_features), self.seq_len
            ),
            ThresholdBlock(StdDevThreshold()),
            PostprocessBlock(TanhNorm()),
            registry=self.registry,
        )
        block_pl.fit(data)

        # Save the model to the registry
        block_pl.save(skeys=["blockpl"], dkeys=["sparsevanillae"])
        _LOGGER.info("Model saved to registry")

        # Train vertex is the last vertex in the pipeline
        return Messages(Message.to_drop())
