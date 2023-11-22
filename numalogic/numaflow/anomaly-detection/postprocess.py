import time
import numpy as np
from typing import List
from orjson import orjson

from pynumaflow.function import Datum, Messages, Message

from src import get_logger
from src.entities import StreamPayload, Header, OutputPayload
from src.tools import WindowScorer
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)


class Postprocess:
    @classmethod
    def postprocess(cls, keys: list[str], payload: StreamPayload) -> (float, dict):
        static_thresh = ConfigManager.get_static_threshold_config(config_name=keys[0])
        postprocess_conf = ConfigManager.get_postprocess_config(config_name=keys[0])
        unified_config = ConfigManager.get_stream_config(config_name=keys[0]).unified_config

        # TODO: Implement weighted average or max strategy for unified anomaly
        # if weights:
        #     weighted_anomalies = np.multiply(scores, unified_weights)
        #     unified_anomaly = float(np.sum(weighted_anomalies) / np.sum(unified_weights))

        # Compute score using static thresholding
        metric_arr = payload.get_data()
        win_scorer = WindowScorer(static_thresh, postprocess_conf)
        if payload.header == Header.STATIC_INFERENCE:
            final_score = win_scorer.get_norm_score(metric_arr)
            _LOGGER.info(
                "%s - Final static threshold score: %s, keys: %s, metrics: %s",
                payload.uuid,
                final_score,
                keys,
                payload.metrics,
            )

        # Compute ensemble score otherwise
        else:
            final_score = win_scorer.get_ensemble_score(metric_arr)
            _LOGGER.info(
                "%s - Final ensemble score: %s, static thresh wt: %s, keys: %s, metrics: %s",
                payload.uuid,
                final_score,
                static_thresh.weight,
                keys,
                payload.metrics,
            )

        # TODO: construct map
        metric_scores = {}
        for metric in payload.metrics:
            metric_scores[metric] = 0

        return final_score, metric_scores

    @classmethod
    def get_unified_anomaly(
            cls, keys: List[str], scores: list[float], payload: StreamPayload
    ) -> float:
        unified_config = ConfigManager.get_stream_config(config_name=keys[0]).unified_config
        unified_weights = unified_config.weights
        if unified_weights:
            weighted_anomalies = np.multiply(scores, unified_weights)
            unified_anomaly = float(np.sum(weighted_anomalies) / np.sum(unified_weights))
            _LOGGER.info(
                "%s - Generating unified anomaly, using unified weights. Unified Anomaly: %s",
                payload.uuid,
                unified_anomaly,
            )
        else:
            unified_anomaly = max(scores)
            _LOGGER.info(
                "%s - Generated unified anomaly, using max strategy. Unified Anomaly: %s",
                payload.uuid,
                unified_anomaly,
            )

        return unified_anomaly

    def run(self, keys: List[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()
        _ = datum.event_time
        _ = datum.watermark

        # Construct payload object
        _in_msg = datum.value.decode("utf-8")
        payload = StreamPayload(**orjson.loads(_in_msg))

        _LOGGER.info("%s - Received Msg: { Keys: %s, Payload: %r }", payload.uuid, keys, payload)

        messages = Messages()

        # Perform postprocess
        final_score, metric_scores = self.postprocess(keys, payload)

        out_payload = OutputPayload(
            timestamp=payload.timestamps[-1],
            unified_anomaly=final_score,
            data=metric_scores,
            metadata=payload.metadata,
        )

        _LOGGER.info("%s - Sending Msg: { Keys: %s, Payload: %s }", payload.uuid, keys, out_payload)
        _LOGGER.debug(
            "%s - Time taken in postprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        messages.append(Message(keys=keys, value=out_payload.to_json()))
        return messages
