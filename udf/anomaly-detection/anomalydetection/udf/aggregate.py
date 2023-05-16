import json
from datetime import datetime
from typing import AsyncIterable, List

from pynumaflow.function import Datum, Metadata, MessageT, MessageTs
from anomalydetection import get_logger

_LOGGER = get_logger(__name__)


async def aggregate(keys: List[str], datums: AsyncIterable[Datum], md: Metadata) -> MessageTs:
    _LOGGER.info("Received Msg: { keys: %s, md: %s}", keys, md)
    timestamp = datetime.timestamp(md.interval_window.end)
    rate_map = {"timestamp": timestamp}

    counter = 0
    count_5xx = 0
    request_count = 0
    async for d in datums:
        json_obj = json.loads(d.value)
        count_5xx = count_5xx + json_obj["count_5xx"]
        request_count = request_count + json_obj["request_count"]
        counter = counter + 1

    if request_count != 0:
        rate_map["error_rate"] = count_5xx / request_count
    else:
        rate_map["error_rate"] = 0

    rate_map["error_count"] = count_5xx / counter

    json_str = json.dumps(rate_map)
    _LOGGER.info("Sending Msg: { keys: %s, value: %s}", keys, json_str)
    return MessageTs(MessageT(str.encode(json_str), keys=keys, event_time=datetime.fromtimestamp(timestamp / 1000)))
