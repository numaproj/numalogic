import logging
import os
from dataclasses import dataclass
import mlflow
from dataclasses_json import dataclass_json
from numalogic.registry import MLflowRegistrar
from numpy._typing import ArrayLike

DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(DIR)[0]
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "ml_steps/resources/train_data/train_data.csv")
TRACKING_URI = "http://mlflow-service.numaflow-system.svc.cluster.local:5000"
LOGGER = logging.getLogger(__name__)


@dataclass_json
@dataclass
class Payload:
    data: ArrayLike = None
    anomaly_score: float = 0.0
    uuid: str = None


def save_model(pl: A) -> None:
    ml_registry = MLflowRegistrar(tracking_uri=TRACKING_URI, artifact_type="pytorch")
    mlflow.start_run()
    ml_registry.save(
        skeys=["ae"], dkeys=["model"], primary_artifact=pl.model, **pl.model_properties
    )
    mlflow.end_run()


def load_model():
    try:
        ml_registry = MLflowRegistrar(tracking_uri=TRACKING_URI)
        artifact_dict = ml_registry.load(skeys=["ae"], dkeys=["model"])
        print(artifact_dict)
        return artifact_dict
    except Exception as ex:
        LOGGER.exception("Error while loading model from MLFlow database: %s", ex)
        return None
