import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import orjson
from numpy.typing import NDArray
from pynumaflow.mapper import Datum, Messages, Message
from sklearn.pipeline import make_pipeline

from numalogic._constants import NUMALOGIC_METRICS
from numalogic.config import PreprocessFactory, RegistryFactory
from numalogic.udfs._logger import configure_logger, log_data_payload_values
from numalogic.registry import LocalLRUCache
from numalogic.tools.types import redis_client_t, artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf
from numalogic.udfs._metrics_utility import _increment_counter
from numalogic.udfs.entities import Status, Header
from numalogic.udfs.tools import (
    make_stream_payload,
    get_df,
    _load_artifact,
    _update_gauge_metric,
    get_trainer_message,
    get_static_thresh_message,
)

# TODO: move to config
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", "10000"))
LOAD_LATEST = os.getenv("LOAD_LATEST", "false").lower() == "true"
METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))
_struct_log = configure_logger()


def _get_updated_metrics(metrics: list, shape: tuple) -> list[str]:
    if shape[1] != len(metrics) and shape[1] == 1:
        metrics = ["-".join(metrics)]
    return metrics


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
        logger = _struct_log.bind(udf_vertex=self._vtx)

        # check message sanity
        try:
            data_payload = orjson.loads(datum.value)
        except (orjson.JSONDecodeError, KeyError):  # catch json decode error only
            logger.exception("Error while decoding input json")
            return Messages(Message.to_drop())

        _stream_conf = self.get_stream_conf(data_payload["config_id"])
        _conf = _stream_conf.ml_pipelines[data_payload.get("pipeline_id", "default")]
        raw_df, timestamps = get_df(data_payload=data_payload, stream_conf=_stream_conf)

        logger = log_data_payload_values(logger, data_payload)

        source = NUMALOGIC_METRICS
        if (
            "numalogic_opex_tags" in data_payload["metadata"]
            and "source" in data_payload["metadata"]["numalogic_opex_tags"]
        ):
            source = data_payload["metadata"]["numalogic_opex_tags"]["source"]

        _metric_label_values = {
            "source": source,
            "vertex": self._vtx,
            "composite_key": ":".join(keys),
            "config_id": data_payload["config_id"],
            "pipeline_id": data_payload.get("pipeline_id", "default"),
        }

        _increment_counter(
            counter="MSG_IN_COUNTER",
            labels=_metric_label_values,
            is_enabled=METRICS_ENABLED,
        )
        # Drop message if dataframe shape conditions are not met
        if raw_df.shape[0] < _stream_conf.window_size or raw_df.shape[1] != len(_conf.metrics):
            logger.critical("Dataframe shape conditions not met ", raw_df_shape=raw_df.shape)
            _increment_counter(
                counter="DATASHAPE_ERROR_COUNTER",
                labels=_metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            _increment_counter(
                counter="MSG_DROPPED_COUNTER",
                labels=_metric_label_values,
                is_enabled=METRICS_ENABLED,
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
                payload = replace(payload, status=Status.ARTIFACT_FOUND)
                logger = logger.bind(artifact_source=preproc_artifact.extras.get("source"))
            else:
                msgs = Messages(get_trainer_message(keys, _stream_conf, payload))
                if _conf.numalogic_conf.score.adjust:
                    msgs.append(get_static_thresh_message(keys, payload))
                logger.exception("Artifact model not loaded!")
                return msgs
        # Model will not be in registry
        else:
            # Load configuration for the config_id
            _increment_counter(
                "SOURCE_COUNTER",
                labels=({"artifact_source": "config"} | _metric_label_values),
                is_enabled=METRICS_ENABLED,
            )
            preproc_clf = self._load_model_from_config(_conf.numalogic_conf.preprocess)
            payload = replace(payload, status=Status.ARTIFACT_FOUND)
        try:
            x_scaled = self.compute(model=preproc_clf, input_=payload.get_data())

            # make metrics list matching same shape as data
            payload = replace(
                payload, metrics=_get_updated_metrics(payload.metrics, x_scaled.shape)
            )

            _update_gauge_metric(x_scaled, payload.metrics, _metric_label_values)
            payload = replace(
                payload,
                data=x_scaled,
                status=Status.ARTIFACT_FOUND,
                header=Header.MODEL_INFERENCE,
            )
            logger.info(
                "Successfully preprocessed!",
                keys=keys,
                payload_metrics=payload.metrics,
                x_scaled=np.array2string(x_scaled),
                execution_time_ms=round((time.perf_counter() - _start_time) * 1000, 4),
            )
        except RuntimeError:
            _increment_counter(
                counter="RUNTIME_ERROR_COUNTER",
                labels=_metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            logger.exception(
                "Runtime preprocess error!",
                status=Status.RUNTIME_ERROR,
                payload_metrics=payload.metrics,
                composite_keys=payload.composite_keys,
            )
            # TODO check again what error is causing this and if retraining is required
            payload = replace(
                payload,
                status=Status.RUNTIME_ERROR,
            )
            msgs = Messages(
                get_trainer_message(keys, _stream_conf, payload, **_metric_label_values),
            )
            if _conf.numalogic_conf.score.adjust:
                msgs.append(get_static_thresh_message(keys, payload))
            return msgs

        _increment_counter(
            counter="MSG_PROCESSED_COUNTER",
            labels=_metric_label_values,
            is_enabled=METRICS_ENABLED,
        )
        return Messages(Message(keys=keys, value=payload.to_json(), tags=["inference"]))

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
        try:
            x_scaled = model.transform(input_)
        except Exception as err:
            raise RuntimeError("Model transform failed!") from err
        return x_scaled
