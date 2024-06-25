import os
import time
from typing import Optional

import orjson
from numpy import typing as npt
from pynumaflow.mapper import Datum, Messages, Message

from numalogic.tools.types import artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf
from numalogic.udfs._logger import configure_logger, log_data_payload_values
from numalogic.udfs._metrics_utility import _increment_counter

METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))

_struct_log = configure_logger()


class PayloadTransformer(NumalogicUDF):
    """
    PayloadGenerator appends pipeline_id to the payload.

    Args:
        pl_conf: PipelineConf instance
    """

    @classmethod
    def compute(cls, model: artifact_t, input_: npt.NDArray[float], **kwargs):
        pass

    def __init__(self, pl_conf: Optional[PipelineConf] = None):
        super().__init__(pl_conf=pl_conf, _vtx="payload_adder")

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        The pipeline function here receives data from the data source.

        Perform ML pipelining on the input data based on the ml_pipelines provided in config

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

        _metric_label_values = {
            "vertex": self._vtx,
            "composite_key": ":".join(keys),
            "config_id": data_payload["config_id"],
        }

        _increment_counter(
            counter="MSG_IN_COUNTER",
            labels=_metric_label_values,
            is_enabled=METRICS_ENABLED,
        )
        # create a new message for each ML pipeline
        messages = Messages()
        for pipeline in _stream_conf.ml_pipelines:
            data_payload["pipeline_id"] = pipeline
            messages.append(Message(keys=keys, value=orjson.dumps(data_payload)))

        logger = log_data_payload_values(logger, data_payload)
        logger.info(
            "Appended pipeline id to the payload",
            keys=keys,
            execution_time_ms=round((time.perf_counter() - _start_time) * 1000, 4),
        )
        _increment_counter(
            counter="MSG_PROCESSED_COUNTER",
            labels=_metric_label_values,
            is_enabled=METRICS_ENABLED,
        )
        return messages
