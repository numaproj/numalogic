import logging
from typing import Optional
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from orjson import orjson
from pynumaflow.function import Datum, Messages, Message
from sklearn.preprocessing import StandardScaler

from numalogic.blocks import (
    BlockPipeline,
    PreprocessBlock,
    NNBlock,
    ThresholdBlock,
    PostprocessBlock,
)
from numalogic.config import NumalogicConf, RedisConf, DataStreamConf
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.numaflow import NumalogicUDF
from numalogic.numaflow.entities import OutputPayload, TrainerPayload
from numalogic.registry import RedisRegistry
from numalogic.registry.redis_registry import get_redis_client_from_conf
from numalogic.tools.exceptions import RedisRegistryError, ModelKeyNotFound, ConfigError
from numalogic.transforms import TanhNorm

_LOGGER = logging.getLogger(__name__)
TRAIN_VTX_KEY = "train"


class InferenceBlockUDF(NumalogicUDF):
    """UDF to preprocess the input data for ML inference."""

    def __init__(
        self, numalogic_conf: NumalogicConf, redis_conf: RedisConf, stream_conf: DataStreamConf
    ):
        super().__init__()
        self.conf = numalogic_conf
        self.stream_conf = stream_conf
        self.model_registry = RedisRegistry(client=self._get_redis_client(redis_conf))
        self._blocks_args = (
            PreprocessBlock(StandardScaler()),
            NNBlock(
                SparseVanillaAE(
                    seq_len=self.stream_conf.window_size, n_features=len(self.stream_conf.metrics)
                ),
                self.stream_conf.window_size,
            ),
            ThresholdBlock(StdDevThreshold()),
            PostprocessBlock(TanhNorm()),
        )
        self._dkeys = ("stdscaler", "sparsevanillae", "stddevthreshold")

    @staticmethod
    def _get_redis_client(redis_conf: RedisConf):
        return get_redis_client_from_conf(redis_conf)

    @staticmethod
    def read_datum(datum: Datum, uuid: str = "") -> dict:
        """Read the input datum and return the data payload."""
        try:
            data_payload = orjson.loads(datum.value)
        except orjson.JSONDecodeError as e:
            _LOGGER.exception("%s - Error while reading input json %r", uuid, e)
            return {}
        _LOGGER.info("%s - Data payload: %s", uuid, data_payload)
        return data_payload

    def extract_data(self, input_data: dict, uuid: str = "") -> Optional[npt.NDArray]:
        """Extract the data from the input payload."""
        stream_array = []
        metrics = self.stream_conf.metrics
        if not metrics:
            raise ConfigError("No metrics found in stream config")
        for row in input_data["data"]:
            stream_array.append([row[metric] for metric in self.stream_conf.metrics])
        if not stream_array:
            _LOGGER.warning("%s - No data found in input payload", uuid)
            return None
        return np.array(stream_array)

    def _construct_output(
        self, anomaly_scores: npt.NDArray, input_payload: dict, model_version: str
    ) -> OutputPayload:
        """Construct the output payload."""
        unified_score = np.max(anomaly_scores)
        metric_data = {
            "metric": {
                "anomaly_score": anomaly_scores[idx],
            }
            for idx, metric in enumerate(self.stream_conf.metrics)
        }
        metric_data["model_version"] = model_version

        return OutputPayload(
            timestamp=input_payload["end_time"],
            unified_anomaly=unified_score,
            data=metric_data,
            metadata=input_payload.get("metadata", {}),
        )

    def _send_training_req(self, keys: list[str], uuid: str = "") -> Messages:
        """Send a training request to the training UDF."""
        train_payload = TrainerPayload(
            uuid=uuid, composite_keys=keys, metrics=self.stream_conf.metrics
        )
        return Messages(
            Message(keys=keys, value=train_payload.to_json(), tags=[TRAIN_VTX_KEY])
        )

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """Runs the UDF."""
        _uuid = uuid4().hex
        input_payload = self.read_datum(datum, uuid=_uuid)
        if not input_payload:
            return Messages(Message.to_drop())

        block_pl = BlockPipeline(*self._blocks_args, registry=self.model_registry)
        stream_array = self.extract_data(input_payload, uuid=_uuid)
        if stream_array is None:
            return Messages(Message.to_drop())

        # Load the model from the registry
        try:
            artifact_data = block_pl.load(skeys=keys, dkeys=self._dkeys)
        except ModelKeyNotFound:
            _LOGGER.info("%s - Model not found for skeys: %s, dkeys: %s", _uuid, keys, self._dkeys)
            return self._send_training_req(keys, _uuid)
        except RedisRegistryError as redis_err:
            _LOGGER.exception("%s - Error loading block pipeline: %r", _uuid, redis_err)
            return Messages(Message.to_drop())

        # Run inference
        try:
            raw_scores = block_pl(stream_array)
        except Exception as err:
            _LOGGER.error("%s - Error running block pipeline: %r", _uuid, err)
            return Messages(Message.to_drop())

        # Find final anomaly score
        anomaly_scores = np.mean(raw_scores, axis=0)
        out_payload = self._construct_output(
            anomaly_scores, input_payload, artifact_data.extras.get("version")
        )
        return Messages(Message(keys=keys, value=out_payload.to_json()))
