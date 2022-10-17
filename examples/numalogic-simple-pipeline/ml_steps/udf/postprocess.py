import json
import logging

from pynumaflow.function import Messages, Message, Datum

from ml_steps.tools import Status
from ml_steps.utlity_function import _load_msg_packet, _convert_packet_payload

LOGGER = logging.getLogger(__name__)


def postprocess(key: str, datum: Datum) -> Messages:
    # Load data and convert bytes to MessagePacket
    msg_packet = _load_msg_packet(datum.value)

    # Postprocess on Inferred data column
    msg_packet.anomaly_score = msg_packet.df["Inferred_value"].mean()

    # Update the message status
    msg_packet.status = Status.POST_PROCESSED.value

    # Convert MessagePacket back to bytes
    payload = _convert_packet_payload(msg_packet)
    LOGGER.info("Postprocess for metrics %s completed.", msg_packet.metric_name)
    messages = Messages()
    messages.append(Message.to_all(json.dumps(payload).encode("utf-8")))
    return messages
