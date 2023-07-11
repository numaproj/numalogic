import json
import logging
import os
from dataclasses import dataclass, asdict

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(DIR)[0]
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "src/resources/train_data.csv")
TRACKING_URI = "http://mlflow-service.default.svc.cluster.local:5000"
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Payload:
    """Payload to be used for inter-vertex data transfer."""

    uuid: str
    arr: list[float]
    anomaly_score: float = None
    is_artifact_valid: bool = True

    def get_array(self) -> npt.NDArray[float]:
        return np.asarray(self.arr)

    def set_array(self, arr: list[float]) -> None:
        self.arr = arr

    def to_json(self) -> bytes:
        """Converts the payload to json."""
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, json_payload: bytes) -> Self:
        """Converts the json payload to Payload object."""
        return cls(**json.loads(json_payload.decode("utf-8")))
