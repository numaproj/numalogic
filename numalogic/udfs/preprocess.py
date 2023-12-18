import logging
import os
import time
from dataclasses import replace
from typing import Optional

import orjson
from numpy.typing import NDArray
from pynumaflow.mapper import Datum, Messages, Message
from sklearn.pipeline import make_pipeline

from numalogic.config import PreprocessFactory, RegistryFactory
from numalogic.udfs._metrics import (
    DATASHAPE_ERROR_COUNTER,
    MSG_DROPPED_COUNTER,
    SOURCE_COUNTER,
    MSG_PROCESSED_COUNTER,
    MSG_IN_COUNTER,
    RUNTIME_ERROR_COUNTER,
    MODEL_STATUS_COUNTER,
    UDF_TIME,
    _increment_counter,
)
from numalogic.registry import LocalLRUCache
from numalogic.tools.types import redis_client_t, artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf
from numalogic.udfs.entities import Status, Header
from numalogic.udfs.tools import make_stream_payload, get_df, _load_artifact

# TODO: move to config
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", "10000"))
LOAD_LATEST = os.getenv("LOAD_LATEST", "false").lower() == "true"

_LOGGER = logging.getLogger(__name__)


class PreprocessUDF(NumalogicUDF):
    """
    Preprocess UDF for Numalogic.

    Args:
        r_client: Redis client
        pl_conf: PipelineConf instance
    """

    __slots__ = ("registry_conf", "model_registry", "preproc_factory")

    def __init__(self, r_client: redis_client_t, pl_conf: Optional[PipelineConf] = None):
        super().__init__(pl_conf=pl_conf, _vtx="preprocess")
        self.registry_conf = self.pl_conf.registry_conf
        model_registry_cls = RegistryFactory.get_cls(self.registry_conf.name)
        self.model_registry = model_registry_cls(
            client=r_client,
            cache_registry=LocalLRUCache(
                ttl=LOCAL_CACHE_TTL,
                cachesize=LOCAL_CACHE_SIZE,
                jitter_sec=self.registry_conf.jitter_conf.jitter_sec,
                jitter_steps_sec=self.registry_conf.jitter_conf.jitter_steps_sec,
            ),
        )
        self.preproc_factory = PreprocessFactory()

    def _load_model_from_config(self, preprocess_cfg):
        preproc_clfs = []
        for _cfg in preprocess_cfg:
            _clf = self.preproc_factory.get_instance(_cfg)
            preproc_clfs.append(_clf)
        return make_pipeline(*preproc_clfs)

    @UDF_TIME.time()
    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        The preprocess function here receives data from the data source.

        Perform preprocess on the input data.

        Args:
        -------
        keys: List of keys
        datum: Datum object

        Returns
        -------
        Messages instance

        """
        _start_time = time.perf_counter()

        # check message sanity
        try:
            data_payload = orjson.loads(datum.value)
            _LOGGER.info("%s - Data payload: %s", data_payload["uuid"], data_payload)
        except (orjson.JSONDecodeError, KeyError):  # catch json decode error only
            _LOGGER.exception("Error while decoding input json")
            return Messages(Message.to_drop())

        _stream_conf = self.get_stream_conf(data_payload["config_id"])
        _conf = _stream_conf.ml_pipelines[data_payload["pipeline_id"]]
        raw_df, timestamps = get_df(data_payload=data_payload, stream_conf=_stream_conf)

        # TODO: Add pipeline id to the ml metrics
        _metric_label_values = (
            self._vtx,
            ":".join(keys),
            data_payload["config_id"],
            data_payload["pipeline_id"],
        )

        _increment_counter(counter=MSG_IN_COUNTER, labels=_metric_label_values)
        # Drop message if dataframe shape conditions are not met
        if raw_df.shape[0] < _stream_conf.window_size or raw_df.shape[1] != len(_conf.metrics):
            _LOGGER.error("Dataframe shape: (%f, %f) error ", raw_df.shape[0], raw_df.shape[1])
            _increment_counter(
                counter=DATASHAPE_ERROR_COUNTER,
                labels=_metric_label_values,
            )
            _increment_counter(
                counter=MSG_DROPPED_COUNTER,
                labels=_metric_label_values,
            )
            return Messages(Message.to_drop())

        # Make StreamPayload object
        payload = make_stream_payload(data_payload, raw_df, timestamps, keys)

        # Check if model will be present in registry
        if any(_cfg.stateful for _cfg in _conf.numalogic_conf.preprocess):
            preproc_artifact, payload = _load_artifact(
                skeys=[
                    _ckey for _, _ckey in zip(_stream_conf.composite_keys, payload.composite_keys)
                ],
                dkeys=[
                    payload.pipeline_id,
                    *[_cfg.name for _cfg in _conf.numalogic_conf.preprocess],
                ],
                payload=payload,
                model_registry=self.model_registry,
                load_latest=LOAD_LATEST,
                vertex=self._vtx,
            )
            if preproc_artifact:
                preproc_clf = preproc_artifact.artifact
                _LOGGER.info(
                    "%s - Loaded model from: %s",
                    payload.uuid,
                    preproc_artifact.extras.get("source"),
                )
                payload = replace(payload, status=Status.ARTIFACT_FOUND)
            else:
                payload = replace(
                    payload, status=Status.ARTIFACT_NOT_FOUND, header=Header.TRAIN_REQUEST
                )
                _increment_counter(
                    counter=MODEL_STATUS_COUNTER,
                    labels=(payload.status.value, *_metric_label_values),
                )
                return Messages(Message(keys=keys, value=payload.to_json()))
        # Model will not be in registry
        else:
            # Load configuration for the config_id
            _LOGGER.info("%s - Initializing model from config: %s", payload.uuid, payload)
            _increment_counter(SOURCE_COUNTER, labels=("config", *_metric_label_values))
            preproc_clf = self._load_model_from_config(_conf.numalogic_conf.preprocess)
            payload = replace(payload, status=Status.ARTIFACT_FOUND)
        try:
            x_scaled = self.compute(model=preproc_clf, input_=payload.get_data())
            payload = replace(
                payload,
                data=x_scaled,
                status=Status.ARTIFACT_FOUND,
                header=Header.MODEL_INFERENCE,
            )
            _LOGGER.info(
                "%s - Successfully preprocessed, Keys: %s, Metrics: %s, x_scaled: %s",
                payload.uuid,
                keys,
                payload.metrics,
                x_scaled,
            )
        except RuntimeError:
            _increment_counter(
                counter=RUNTIME_ERROR_COUNTER,
                labels=_metric_label_values,
            )
            _LOGGER.exception(
                "%s - Runtime inference error! Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            # TODO check again what error is causing this and if retraining is required
            payload = replace(payload, status=Status.RUNTIME_ERROR, header=Header.TRAIN_REQUEST)
            _increment_counter(
                counter=MODEL_STATUS_COUNTER,
                labels=(
                    payload.status.value,
                    *_metric_label_values,
                ),
            )
            return Messages(Message(keys=keys, value=payload.to_json()))
        _LOGGER.debug(
            "%s - Time taken to execute Preprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        _increment_counter(
            counter=MODEL_STATUS_COUNTER,
            labels=(
                payload.status.value,
                *_metric_label_values,
            ),
        )
        _increment_counter(
            counter=MSG_PROCESSED_COUNTER,
            labels=_metric_label_values,
        )
        return Messages(Message(keys=keys, value=payload.to_json()))

    @classmethod
    def compute(
        cls, model: artifact_t, input_: Optional[NDArray[float]] = None, **_
    ) -> NDArray[float]:
        """
        Perform inference on the input data.

        Args:
            model: Model artifact
            input_: Input data

        Returns
        -------
            Preprocessed array

        Raises
        ------
            RuntimeError: If preprocess fails
        """
        _start_time = time.perf_counter()
        try:
            x_scaled = model.transform(input_)
            _LOGGER.info("Time taken in preprocessing: %.4f sec", time.perf_counter() - _start_time)
        except Exception as err:
            raise RuntimeError("Model transform failed!") from err
        else:
            return x_scaled
