import logging

import numpy as np
from numalogic.udfs import NumalogicUDF
from numalogic.transforms import TanhNorm
from pynumaflow.function import Messages, Message, Datum

from src.utils import Payload

LOGGER = logging.getLogger(__name__)


class Postprocess(NumalogicUDF):
    """UDF to postprocess the anomaly score, and scale it between [0,10]."""

    def __init__(self):
        super().__init__()

    def exec(self, _: list[str], datum: Datum) -> Messages:
        """The postprocess transforms the inferred data into anomaly score between [0,10]
        and sends it to log sink.

        For more information about the arguments, refer:
        https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py
        """
        # Load json data
        payload = Payload.from_json(datum.value)

        # Postprocess step
        data = payload.get_array()

        # Taking mean of the anomaly scores
        normalizer = TanhNorm()
        payload.anomaly_score = normalizer.fit_transform(np.mean(data))

        LOGGER.info("%s - The anomaly score is: %s", payload.uuid, payload.anomaly_score)

        # Convert Payload back to bytes
        return Messages(Message(value=payload.to_json()))
