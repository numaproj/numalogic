import time
from typing import List

from pynumaflow.function import Messages, Datum

from src import get_logger

_LOGGER = get_logger(__name__)

class Train:

    def run(self, keys: List[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()
        _ = datum.event_time
        _ = datum.watermark
        _LOGGER.info("Received Msg: { Keys: %s, Payload: %s }", keys, datum.value.decode("utf-8"))
        # Construct payload object
        msgs = datum.value.decode("utf-8")

        for msg in msgs:
            print(msg)

        return Messages()


