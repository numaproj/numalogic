from dataclasses import dataclass
from enum import Enum

from dataclasses_json import dataclass_json


class MessagePacket:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.df = None
        self.status = None
        self.anomaly_score = 0.0


@dataclass_json
@dataclass
class Metric:
    timestamp: str
    value: float = 0


class Status(str, Enum):
    RAW = "raw"
    EXTRACTED = "extracted"
    PRE_PROCESSED = "pre_processed"
    INFERRED = "inferred"
    POST_PROCESSED = "post_processed"
    TRAIN = "train"
