import logging
import time
from typing import Optional

import orjson
from numpy import typing as npt
from pynumaflow.mapper import Datum, Messages, Message

from numalogic.tools.types import artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf

_LOGGER = logging.getLogger(__name__)


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

        # check message sanity
        try:
            data_payload = orjson.loads(datum.value)
            _LOGGER.info("%s - Data payload: %s", data_payload["uuid"], data_payload)
        except (orjson.JSONDecodeError, KeyError):  # catch json decode error only
            _LOGGER.exception("Error while decoding input json")
            return Messages(Message.to_drop())

        _stream_conf = self.get_stream_conf(data_payload["config_id"])

        # create a new message for each ML pipeline
        messages = Messages()
        for pipeline in _stream_conf.ml_pipelines:
            data_payload["pipeline_id"] = pipeline
            messages.append(Message(keys=keys, value=orjson.dumps(data_payload)))

        _LOGGER.debug(
            "Time taken to execute Pipeline: %.4f sec",
            time.perf_counter() - _start_time,
        )
        return messages
