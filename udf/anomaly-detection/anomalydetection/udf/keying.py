import json
from typing import List

from pynumaflow.function import Messages, Message, Datum, MessageT

from anomalydetection import get_logger

_LOGGER = get_logger(__name__)


def keying(keys: List[str], datum: Datum) -> Messages:
    _ = datum.event_time
    _ = datum.watermark
    messages = Messages()
    _LOGGER.info("Received Msg: { keys: %s, value: %s }", datum.value, keys)

    try:
        json_obj = json.loads(datum.value)
    except Exception as e:
        _LOGGER.error("Error while reading input json %r", e)
        messages.append(Message.to_drop())
        return messages

    source_id = json_obj["source_asset_id"]
    destination_id = json_obj["destination_asset_id"]
    env = json_obj["env"]

    if source_id is None or destination_id is None or env is None:
        messages.append(Message.to_drop())
        return messages

    messages.append(Message(datum.value, keys=[destination_id, env], tags=["service-mesh"]))
    messages.append(Message(datum.value, keys=[source_id, destination_id, env], tags=["service-mesh-s2s"]))
    _LOGGER.info("Sending Msgs: { keys: [%s, %s] value: %s }", [destination_id, env],
                 [source_id, destination_id, env], datum.value)
    return messages
