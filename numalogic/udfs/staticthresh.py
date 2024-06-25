import os

from orjson import orjson
from pynumaflow.mapper import Datum, Messages, Message

from numalogic.config import AggregatorFactory, ScoreAdjustConf, AggregatorConf
from numalogic.models.threshold import SigmoidThreshold
from numalogic.tools.aggregators import aggregate_window, aggregate_features
from numalogic.udfs import NumalogicUDF, PipelineConf
import numpy.typing as npt

from numalogic.udfs._logger import configure_logger, log_data_payload_values
from numalogic.udfs._metrics_utility import _increment_counter
from numalogic.udfs.entities import StreamPayload, OutputPayload
from numalogic.udfs.tools import _update_gauge_metric

METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))
SCORE_PREFIX = os.getenv("SCORE_PREFIX", "unified")
_struct_log = configure_logger()


class StaticThresholdUDF(NumalogicUDF):
    """
    Static thresholding UDF, which computes the static anomaly scores.

    Args:
        pl_conf: PipelineConf object
    """

    def __init__(self, pl_conf: PipelineConf, **_):
        super().__init__(pl_conf=pl_conf, _vtx="staticthresh")

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        Processes the input data and computes the static anomaly scores.

        Args:
        -------
        keys: List of keys
        datum: Datum object.

        Returns
        -------
        Messages instance
        """
        json_data_payload = orjson.loads(datum.value)
        payload = StreamPayload(**json_data_payload)
        conf = self.get_ml_pipeline_conf(payload.config_id, payload.pipeline_id)
        adjust_conf = conf.numalogic_conf.score.adjust
        _metric_label_values = {
            "source": ":".join(payload.composite_keys),
            "vertex": self._vtx,
            "composite_key": ":".join(payload.composite_keys),
            "config_id": payload.config_id,
            "pipeline_id": payload.pipeline_id,
        }
        _increment_counter(
            counter="MSG_IN_COUNTER",
            labels=_metric_label_values,
            is_enabled=METRICS_ENABLED,
        )
        logger = _struct_log.bind(udf_vertex=self._vtx)
        logger = log_data_payload_values(logger, json_data_payload)

        if not adjust_conf:
            logger.warning("No score adjust config found")
            return Messages(Message.to_drop())

        try:
            y_features = self.compute(
                input_=payload.get_data(original=True, metrics=list(adjust_conf.upper_limits)),
                adjust_conf=adjust_conf,
            )
            _update_gauge_metric(y_features, list(adjust_conf.upper_limits), _metric_label_values)
            y_unified = self.compute_unified_score(y_features, adjust_conf.feature_agg)
        except RuntimeError:
            logger.exception("Error occurred while computing static anomaly scores")
            _increment_counter(
                "RUNTIME_ERROR_COUNTER", _metric_label_values, is_enabled=METRICS_ENABLED
            )
            return Messages(Message.to_drop())

        out_payload = OutputPayload(
            uuid=payload.uuid,
            config_id=payload.config_id,
            pipeline_id=payload.pipeline_id,
            composite_keys=payload.composite_keys,
            timestamp=payload.end_ts,
            unified_anomaly=y_unified,
            data=self._additional_scores(adjust_conf, y_features, y_unified),
            metadata=payload.metadata,
        )
        logger.info(
            "Sending output payload",
            keys=out_payload.composite_keys,
            y_unified=y_unified,
            y_features=y_features,
        )
        _increment_counter(
            counter="MSG_PROCESSED_COUNTER",
            labels=_metric_label_values,
            is_enabled=METRICS_ENABLED,
        )
        return Messages(Message(keys=keys, value=out_payload.to_json(), tags=["output"]))

    @staticmethod
    def _additional_scores(
        adjust_conf: ScoreAdjustConf, y_features: npt.NDArray[float], y_unified: float
    ) -> dict[str, float]:
        """
        Additional scores to be computed.

        Args:
        -------
            feat_names: List of feature names
            y_features: Anomaly scores
            y_unified: Unified anomaly score

        Returns
        -------
            Additional scores
        """
        scores_payload = {f"{SCORE_PREFIX}_ST": y_unified}
        _feat_names = [f"{f}_ST" for f in adjust_conf.upper_limits]
        scores_payload |= dict(zip(_feat_names, y_features))
        return scores_payload

    @classmethod
    def compute(
        cls, input_: npt.NDArray[float], adjust_conf: ScoreAdjustConf, **_
    ) -> npt.NDArray[float]:
        """
        Compute static thresholding over the raw input features.

        Args:
            input_: Input data
            adjust_conf: Score adjust Config

        Returns
        -------
        npt.NDArray[float]
        """
        scorer = SigmoidThreshold(*adjust_conf.upper_limits.values())
        try:
            return cls.compute_feature_scores(scorer.score_samples(input_), adjust_conf.window_agg)
        except Exception as err:
            raise RuntimeError("Static Thresholding failed!") from err

    @classmethod
    def compute_feature_scores(
        cls, scores: npt.NDArray[float], win_agg_conf: AggregatorConf
    ) -> npt.NDArray[float]:
        """
        Aggregate scores over the window length.

        Args:
        -------
            scores: anomaly scores (Shape: seq_len x n_features)
            win_agg_conf: Window aggregator Config

        Returns
        -------
            Aggregated scores of shape (n_features,)
        """
        return aggregate_window(
            scores,
            agg_func=AggregatorFactory.get_func(win_agg_conf.method),
            **win_agg_conf.conf,
        )

    @classmethod
    def compute_unified_score(
        cls, scores: npt.NDArray[float], feat_agg_conf: AggregatorConf
    ) -> float:
        """
        Aggregate scores over the features to get a unified score.

        Args:
        -------
            scores: anomaly scores (Shape: n_features, )
            feat_agg_conf: Feature aggregator Config

        Returns
        -------
            Unified score (float)
        """
        try:
            return aggregate_features(
                scores.reshape(1, -1),
                agg_func=AggregatorFactory.get_func(feat_agg_conf.method),
                **feat_agg_conf.conf,
            ).item()
        except Exception as err:
            raise RuntimeError("Unified Score computation failed!") from err
