import json
import logging
import os

from numalogic.registry import MLflowRegistrar
from pynumaflow.function import Messages, Message, Datum

from ml_steps.pipeline import SimpleMLPipeline
from ml_steps.tools import Status
from ml_steps.utlity_function import _convert_packet_payload, _load_msg_packet

LOGGER = logging.getLogger(__name__)
DEFAULT_WIN_SIZE = 12
DEFAULT_MODEL_NAME = "ae_sparse"
TRACKING_URI = "http://mlflow-service.numaflow-system.svc.cluster.local:5000"


def load_model(payload):
    try:
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        ml_registry = MLflowRegistrar(tracking_uri=TRACKING_URI)
        artifact_dict = ml_registry.load(
            skeys=[payload.metric_name], dkeys=[model_name]
        )
        LOGGER.info("Loaded artifacts for: %s", payload.metric_name)
        return artifact_dict
    except Exception as ex:
        LOGGER.exception("Error while loading model from MLflow database")
        return None


def inference(key: str, datum: Datum):
    # Load data and convert bytes to MessagePacket
    msg_packet = _load_msg_packet(datum.value)

    # Load the model for the given metric name
    artifact_dict = load_model(msg_packet)
    messages = Messages()

    # Check if model exists
    if artifact_dict:
        LOGGER.info(
            "Checking if artifacts for the metrics %s exists", msg_packet.metric_name
        )
        ml_pipeline = SimpleMLPipeline(
            metric=msg_packet.metric_name,
            model_plname=DEFAULT_MODEL_NAME,
            seq_len=DEFAULT_WIN_SIZE,
            model=artifact_dict["primary_artifact"],
        )

        # Load model to the pipeline
        ml_pipeline.load_model(
            path_or_buf=None,
            model=artifact_dict["primary_artifact"],
            **artifact_dict["metadata"]
        )

        # Do the inference
        msg_packet.df["Inferred_value"] = ml_pipeline.infer(
            msg_packet.df["Preprocess_value"].to_numpy().reshape(-1, 1)
        )

        # Update the status to Inferred
        msg_packet.status = Status.INFERRED.value

        # Convert MessagePacket back to bytes and forward it to postprocess vertex
        payload = _convert_packet_payload(msg_packet)
        LOGGER.info("Model inference done. Sending to postprocess step")
        messages.append(
            Message.to_vtx(key="postprocess", value=json.dumps(payload).encode("utf-8"))
        )
    else:

        # Model does not exist. Send it to trainer for training.
        LOGGER.info(
            "Model for metrics %s does not exists. Sending to trainer for training",
            msg_packet.metric_name,
        )

        # Update the message packet to train
        msg_packet.status = Status.TRAIN.value
        payload = _convert_packet_payload(msg_packet)

        # Convert MessagePacket back to bytes and forward it to train vertex
        messages.append(
            Message.to_vtx(key="train", value=json.dumps(payload).encode("utf-8"))
        )
    return messages
