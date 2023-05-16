from dataclasses import dataclass
from enum import Enum
from typing import List, Union, TypeVar
import numpy as np
import numpy.typing as npt
import orjson

Vector = List[float]
Matrix = Union[Vector, List[Vector], npt.NDArray[float]]


class Status(str, Enum):
    RAW = "raw"
    EXTRACTED = "extracted"
    PRE_PROCESSED = "pre_processed"
    INFERRED = "inferred"
    THRESHOLD = "threshold_complete"
    POST_PROCESSED = "post_processed"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_STALE = "artifact_is_stale"
    RUNTIME_ERROR = "runtime_error"


class Header(str, Enum):
    STATIC_INFERENCE = "static_threshold"
    MODEL_INFERENCE = "model_inference"
    TRAIN_REQUEST = "request_training"
    MODEL_STALE = "model_stale"


@dataclass
class _BasePayload:
    uuid: str
    unique_key: str

    @property
    def start_ts(self) -> str:
        return self.unique_key.split()[0]


PayloadType = TypeVar("PayloadType", bound=_BasePayload)


@dataclass
class TrainerPayload(_BasePayload):
    header: Header = Header.TRAIN_REQUEST


@dataclass
class WindowPayload:
    timestamps: List[str]
    metrics: List[str]
    data: Matrix
    raw_data: Matrix

    def get_data(self, original=False) -> npt.NDArray[float]:
        if original:
            return np.asarray(self.raw_data)
        return np.asarray(self.data)

    def set_data(self, arr: Matrix) -> None:
        self.data = arr

    def to_string(self) -> str:
        return f"{{ timestamps: {self.timestamps}, metrics: {self.metrics}, data: {list(self.data)}, raw_data: {list(self.raw_data)}}}"


@dataclass(repr=False)
class StreamPayload(_BasePayload):
    window: WindowPayload
    status: Status = Status.RAW
    header: Header = Header.MODEL_INFERENCE

    @property
    def start_ts(self) -> str:
        return self.window.timestamps[0]

    @property
    def end_ts(self) -> str:
        return self.window.timestamps[-1]

    def set_status(self, status: Status) -> None:
        self.status = status

    def set_header(self, header: Header) -> None:
        self.header = header

    def __repr__(self) -> str:
        return "{ unique_key: %s, header: %s,  window: %s }" % (
            self.unique_key,
            self.header,
            self.window.to_string()
        )


class PayloadFactory:
    __HEADER_MAP = {
        Header.MODEL_INFERENCE: StreamPayload,
        Header.TRAIN_REQUEST: TrainerPayload,
        Header.STATIC_INFERENCE: StreamPayload,
        Header.MODEL_STALE: StreamPayload,
    }

    @classmethod
    def from_json(cls, json_data: Union[bytes, str]) -> PayloadType:
        data = orjson.loads(json_data)
        header = data.get("header")
        if not header:
            raise RuntimeError(f"Header not present in json: {json_data}")
        payload_cls = cls.__HEADER_MAP[header]
        return payload_cls(**data)
