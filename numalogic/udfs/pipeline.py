import json
import os
import time
import orjson
import logging
from typing import Optional

from numpy import typing as npt
from pynumaflow.mapper import Datum, Messages, Message

from numalogic.tools.types import artifact_t
from numalogic.udfs._metrics import UDF_TIME
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf

# TODO: move to config
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", "10000"))
LOAD_LATEST = os.getenv("LOAD_LATEST", "false").lower() == "true"

_LOGGER = logging.getLogger(__name__)


class PipelineUDF(NumalogicUDF):
    """
    Pipeline UDF for Numalogic.

    Args:
        pl_conf: PipelineConf instance
    """

    @classmethod
    def compute(cls, model: artifact_t, input_: npt.NDArray[float], **kwargs):
        pass

    def __init__(self, pl_conf: Optional[PipelineConf] = None):
        super().__init__(pl_conf=pl_conf, _vtx="pipeline")

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

        # create a new message for each ML pipeline
        messages = Messages()
        for pipeline in _stream_conf.ml_pipelines:
            data_payload["pipeline_id"] = pipeline
            messages.append(Message(keys=keys, value=str.encode(json.dumps(data_payload))))

        _LOGGER.debug(
            "Time taken to execute Pipeline: %.4f sec",
            time.perf_counter() - _start_time,
        )
        return messages
