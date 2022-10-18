import os
from dataclasses import dataclass
from enum import Enum
from dataclasses_json import dataclass_json
from numpy._typing import ArrayLike

DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(DIR)[0]
MODEL_PATH = os.path.join(ROOT_DIR, "ml_steps/resources/ml_model/model.pth")
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "ml_steps/resources/train_data/train_data.csv")


@dataclass_json
@dataclass
class Payload:
    data: ArrayLike = None
    status: str = None
    anomaly_score: float = 0.0
    uuid: str = None


# 1. have numpy instead of df
# 2 . mlflow remove...use torch.save


@dataclass_json
@dataclass
class Metric:
    value: float = 0


class Status(str, Enum):
    RAW = "raw"
    EXTRACTED = "extracted"
    PRE_PROCESSED = "pre_processed"
    INFERRED = "inferred"
    POST_PROCESSED = "post_processed"
    TRAIN = "train"
