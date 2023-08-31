import logging

from numalogic.udfs import NumalogicUDF
from numalogic.registry import MLflowRegistry
from pynumaflow.function import Messages, Message, Datum

from src.utils import Payload

LOGGER = logging.getLogger(__name__)
TRACKING_URI = "http://mlflow-service.default.svc.cluster.local:5000"


class Threshold(NumalogicUDF):
    """UDF to apply thresholding to the reconstruction error returned by the autoencoder."""

    def __init__(self):
        super().__init__()
        self.registry = MLflowRegistry(tracking_uri=TRACKING_URI)

    @staticmethod
    def _handle_not_found(payload: Payload) -> Messages:
        """
        Handles the case when the model is not found.
        If model not found, send it to trainer for training.
        """
        LOGGER.warning("%s - Model not found. Training the model.", payload.uuid)

        # Convert Payload back to bytes and conditional forward to train vertex
        payload.is_artifact_valid = False
        return Messages(Message(keys=["train"], value=payload.to_json()))

    def exec(self, _: list[str], datum: Datum) -> Messages:
        """
        UDF that applies thresholding to the reconstruction error returned by the autoencoder.

        For more information about the arguments, refer:
        https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py
        """
        # Load data and convert bytes to Payload
        payload = Payload.from_json(datum.value)

        # Load the threshold model from registry
        thresh_clf_artifact = self.registry.load(
            skeys=["thresh_clf"], dkeys=["model"], artifact_type="sklearn"
        )
        recon_err = payload.get_array().reshape(-1, 1)

        # Check if model exists for inference
        if (not thresh_clf_artifact) or (not payload.is_artifact_valid):
            return self._handle_not_found(payload)

        thresh_clf = thresh_clf_artifact.artifact
        payload.set_array(thresh_clf.predict(recon_err).tolist())

        LOGGER.info("%s - Thresholding complete", payload.uuid)

        # Convert Payload back to bytes and conditional forward to postprocess vertex
        return Messages(Message(keys=["postprocess"], value=payload.to_json()))
