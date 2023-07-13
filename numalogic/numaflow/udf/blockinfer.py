import logging
from datetime import datetime
from typing import Optional

import fakeredis
import numpy as np
import numpy.typing as npt
from orjson import orjson
from pynumaflow.function import Datum, Messages, Message, DatumMetadata
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
from numalogic.numaflow.entities import OutputPayload
from numalogic.registry import RedisRegistry
from numalogic.tools.exceptions import RedisRegistryError, ModelKeyNotFound
from numalogic.transforms import TanhNorm

_LOGGER = logging.getLogger(__name__)
server = fakeredis.FakeServer()
fake_redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)


class InferenceBlock(NumalogicUDF):
    """UDF to preprocess the input data for ML inference."""

    def __init__(
        self, numalogic_conf: NumalogicConf, redis_conf: RedisConf, stream_conf: DataStreamConf
    ):
        super().__init__()
        self.conf = numalogic_conf
        self.stream_conf = stream_conf
        # self.model_registry = RedisRegistry(client=get_redis_client_from_conf(redis_conf))
        self.model_registry = RedisRegistry(client=fake_redis_client)
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
    def read_datum(datum: Datum) -> dict:
        try:
            data_payload = orjson.loads(datum.value)
        except orjson.JSONDecodeError as e:
            _LOGGER.exception("%s - Error while reading input json %r", e)
            return {}
        _LOGGER.info("Data payload: %s", data_payload)
        return data_payload

    def extract_data(self, input_data: dict) -> Optional[npt.NDArray]:
        stream_array = []

        for row in input_data["data"]:
            stream_array.append([row[metric] for metric in self.stream_conf.metrics])
        if not stream_array:
            _LOGGER.warning("No data found in input payload")
            return None
        return np.array(stream_array)

    def _construct_output(
            self, anomaly_scores: npt.NDArray, input_payload: dict, model_version: str
    ) -> OutputPayload:
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

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        input_payload = self.read_datum(datum)
        if not input_payload:
            return Messages(Message.to_drop())

        block_pl = BlockPipeline(*self._blocks_args, registry=self.model_registry)
        stream_array = self.extract_data(input_payload)
        if stream_array is None:
            return Messages(Message.to_drop())

        # Load the model from the registry
        try:
            artifact_data = block_pl.load(skeys=keys, dkeys=self._dkeys)
        except ModelKeyNotFound:
            _LOGGER.info("Model not found for skeys: %s, dkeys: %s", keys, self._dkeys)
            # TODO fix this for training
            return Messages(Message(value=b"", tags=["train"]))

        except RedisRegistryError as redis_err:
            _LOGGER.exception("Error loading block pipeline: %r", redis_err)
            return Messages(Message.to_drop())

        # Run inference
        try:
            raw_scores = block_pl(stream_array)
        except Exception as err:
            _LOGGER.error("Error running block pipeline: %r", err)
            return Messages(Message.to_drop())

        # Find final anomaly score
        anomaly_scores = np.mean(raw_scores, axis=0)
        out_payload = self._construct_output(
            anomaly_scores, input_payload, artifact_data.extras.get("version")
        )
        return Messages(Message(keys=keys, value=out_payload.to_json()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    block_pl = BlockPipeline(
        PreprocessBlock(StandardScaler()),
        NNBlock(SparseVanillaAE(seq_len=12, n_features=3), 12),
        ThresholdBlock(StdDevThreshold()),
        PostprocessBlock(TanhNorm()),
        registry=RedisRegistry(client=fake_redis_client),
    )
    rng = np.random.default_rng()
    block_pl.fit(rng.random((1000, 3)), nn__max_epochs=10)
    skeys = ["fciPluginAsset", "1804835142986662399"]
    block_pl.save(skeys=skeys, dkeys=("stdscaler", "sparsevanillae", "stddevthreshold"))

    udf = InferenceBlock(
        NumalogicConf(),
        RedisConf("host", 6379),
        DataStreamConf(metrics=["degraded", "failed", "success"]),
    )
    dat_value = {
        "data": [
            {"degraded": 0, "failed": 0, "success": 14, "timestamp": 1689189540000},
            {"degraded": 0, "failed": 0, "success": 8, "timestamp": 1689189600000},
            {"degraded": 0, "failed": 0, "success": 6, "timestamp": 1689189660000},
            {"degraded": 0, "failed": 4, "success": 2, "timestamp": 1689189720000},
            {"degraded": 0, "failed": 0, "success": 2, "timestamp": 1689189780000},
            {"degraded": 2, "failed": 0, "success": 4, "timestamp": 1689189840000},
            {"degraded": 0, "failed": 0, "success": 6, "timestamp": 1689190080000},
            {"degraded": 0, "failed": 0, "success": 6, "timestamp": 1689190140000},
            {"degraded": 1, "failed": 0, "success": 2, "timestamp": 1689190500000},
            {"degraded": 0, "failed": 0, "success": 2, "timestamp": 1689190560000},
            {"degraded": 0, "failed": 0, "success": 2, "timestamp": 1689190620000},
            {"degraded": 0, "failed": 0, "success": 2, "timestamp": 1689190680000},
        ],
        "start_time": 1689189540000,
        "end_time": 1689190740000,
        "metadata": {
            "current_pipeline_name": "fciPluginAsset",
            "druid": {"source": "numalogic"},
            "wavefront": {"source": "numalogic"},
        },
    }
    udf(
        skeys,
        Datum(
            keys=skeys,
            value=orjson.dumps(dat_value),
            event_time=datetime.now(),
            watermark=datetime.now(),
            metadata=DatumMetadata(msg_id="", num_delivered=0),
        ),
    )
