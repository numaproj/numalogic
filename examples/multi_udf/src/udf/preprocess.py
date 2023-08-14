import json
import logging
import uuid

from numalogic.transforms import LogTransformer
from numalogic.udfs import NumalogicUDF
from pynumaflow.function import Messages, Message, Datum

from src.utils import Payload

LOGGER = logging.getLogger(__name__)


class Preprocess(NumalogicUDF):
    """UDF to preprocess the input data for ML inference."""

    def __init__(self):
        super().__init__()

    def exec(self, _: list[str], datum: Datum) -> Messages:
        """The preprocess function here transforms the input data for ML inference and sends
        the payload to inference vertex.

        For more information about the arguments, refer:
        https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py
        """
        # Load json data
        series = json.loads(datum.value)["data"]
        payload = Payload(uuid=str(uuid.uuid4()), arr=series)

        # preprocess step
        data = payload.get_array()
        clf = LogTransformer()
        out = clf.fit_transform(data)
        payload.set_array(out.tolist())
        LOGGER.info("%s - Preprocess complete for data: %s", payload.uuid, payload.arr)

        # Return as a Messages object
        return Messages(Message(value=payload.to_json()))
