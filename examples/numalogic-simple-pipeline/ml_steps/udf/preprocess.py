import json
import logging

from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Messages, Message, Datum

from ml_steps.pipeline import SimpleMLPipeline
from ml_steps.tools import Status
from ml_steps.utlity_function import (
    _convert_packet_payload,
    _load_msg_packet,
)

LOGGER = logging.getLogger(__name__)

PRE_PROC = LogTransformer()


def preprocess(key: str, datum: Datum) -> Messages:

    # Load data and convert bytes to MessagePacket
    msg_packet = _load_msg_packet(datum.value)

    ml_pipeline = SimpleMLPipeline(
        metric=msg_packet.metric_name, preprocess_steps=[PRE_PROC]
    )

    # Clean and preprocess the data
    msg_packet.df = ml_pipeline.clean_data(msg_packet.df)
    msg_packet.df["Preprocess_value"] = ml_pipeline.preprocess(
        msg_packet.df["Value"], train=False
    )

    # Update the message status
    msg_packet.status = Status.PRE_PROCESSED.value
    LOGGER.info("Preprocess for metrics %s completed.", msg_packet.metric_name)

    # Convert MessagePacket back to bytes
    payload = _convert_packet_payload(msg_packet)
    messages = Messages()
    messages.append(Message.to_all(json.dumps(payload).encode("utf-8")))
    return messages
