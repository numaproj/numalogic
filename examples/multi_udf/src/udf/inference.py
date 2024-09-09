import logging
import os

import numpy.typing as npt

from numalogic.connectors import RedisConf
from numalogic.connectors.redis import get_redis_client, get_redis_client_from_conf
from numalogic.models.autoencoder import TimeseriesTrainer
from numalogic.udfs import NumalogicUDF
from numalogic.registry import MLflowRegistry, ArtifactData, RedisRegistry
from numalogic.tools.data import StreamingDataset
from pynumaflow.function import Messages, Message, Datum
from torch.utils.data import DataLoader

from src.utils import Payload

LOGGER = logging.getLogger(__name__)
WIN_SIZE = int(os.getenv("WIN_SIZE"))
REGISTRY = os.getenv("REGISTRY", "mlflow")
TRACKING_URI = "http://mlflow-service.default.svc.cluster.local:5000"
REDIS_URI = "isbsvc-redis-isbs-redis-svc.default.svc"


class Inference(NumalogicUDF):
    """
    The inference function here performs inference on the streaming data and sends
    the payload to threshold vertex.
    """

    def __init__(self):
        super().__init__()
        if REGISTRY == "mlflow":
            self.registry = MLflowRegistry(tracking_uri=TRACKING_URI)
        if REGISTRY == "redis":
            redis_conf = RedisConf(url=REDIS_URI, port=26379)
            self.registry = get_redis_client_from_conf(redis_conf)

    def load_model(self) -> ArtifactData:
        """Loads the model from the registry."""
        return self.registry.load(skeys=["ae"], dkeys=["model"])

    @staticmethod
    def _infer(artifact_data: ArtifactData, stream_data: npt.NDArray[float]) -> list[float]:
        """Performs inference on the streaming data."""
        main_model = artifact_data.artifact
        streamloader = DataLoader(StreamingDataset(stream_data, WIN_SIZE))

        trainer = TimeseriesTrainer()
        reconerr = trainer.predict(main_model, dataloaders=streamloader)
        return reconerr.tolist()

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        Here inference is done on the data, given, the ML model is present
        in the registry. If a model does not exist, the payload is flagged for training.
        It then passes to the threshold vertex.

        For more information about the arguments, refer:
        https://github.com/numaproj/numaflow-python/blob/main/pynumaflow/function/_dtypes.py
        """
        # Load data and convert bytes to Payload
        payload = Payload.from_json(datum.value)

        artifact_data = self.load_model()
        stream_data = payload.get_array().reshape(-1, 1)

        # Check if model exists for inference
        if artifact_data:
            payload.set_array(self._infer(artifact_data, stream_data))
            LOGGER.info("%s - Inference complete", payload.uuid)
        else:
            # If model not found, set status as not found
            LOGGER.warning("%s - Model not found", payload.uuid)
            payload.is_artifact_valid = False

        # Convert Payload back to bytes and conditional forward to threshold vertex
        return Messages(Message(value=payload.to_json()))
