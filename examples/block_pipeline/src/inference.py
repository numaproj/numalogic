import json
import logging

import numpy as np
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
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.transforms import TanhNorm
from pynumaflow.function import Messages, Datum, Message
from sklearn.preprocessing import StandardScaler

from src.utils import RedisClient

_LOGGER = logging.getLogger(__name__)


class Inference(NumalogicUDF):
    """UDF to preprocess the input data for ML inference."""

    def __init__(self, seq_len: int = 12, num_series: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = num_series
        self.registry = RedisRegistry(client=RedisClient().get_client())

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        # Load json data
        series = json.loads(datum.value)["data"]

        block_pl = BlockPipeline(
            PreprocessBlock(StandardScaler()),
            NNBlock(
                SparseVanillaAE(seq_len=self.seq_len, n_features=self.n_features), self.seq_len
            ),
            ThresholdBlock(StdDevThreshold()),
            PostprocessBlock(TanhNorm()),
            registry=self.registry,
        )

        # Load the model from the registry
        try:
            block_pl.load(skeys=["blockpl"], dkeys=["sparsevanillae"])
        except RedisRegistryError as warn:
            _LOGGER.warning("Error loading block pipeline: %r", warn)
            return Messages(Message(value=b"", tags=["train"]))

        # Run inference
        try:
            output = block_pl(np.asarray(series).reshape(-1, self.n_features))
        except Exception:
            _LOGGER.exception("Error running block pipeline")
            return Messages(Message.to_drop())

        anomaly_score = np.mean(output)
        return Messages(Message(tags=["out"], value=json.dumps({"score": anomaly_score}).encode()))
