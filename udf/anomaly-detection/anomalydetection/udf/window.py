import json
from datetime import datetime
from typing import AsyncIterable, List

from pynumaflow.function import MessageTs, MessageT, Datum, Metadata

from anomalydetection import get_logger

_LOGGER = get_logger(__name__)


async def window(keys: List[str], datums: AsyncIterable[Datum], md: Metadata) -> MessageTs:
    _LOGGER.info("Received Msg: { keys: %s, md: %s}", keys, md)
    output = []
    timestamp = datetime.timestamp(md.interval_window.end)

    async for d in datums:
        json_obj = json.loads(d.value)
        output.append(json_obj)

    json_str = json.dumps(output)
    _LOGGER.info("Sending Msg: { keys: %s, value: %s }", keys, json_str)
    return MessageTs(MessageT(str.encode(json_str), keys=keys, event_time=datetime.fromtimestamp(timestamp/1000)))
