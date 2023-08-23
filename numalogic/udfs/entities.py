from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union, TypeVar, Optional

import numpy as np
import numpy.typing as npt
import orjson

Vector = list[float]
Matrix = Union[Vector, list[Vector], npt.ArrayLike]


class Status(str, Enum):
    ARTIFACT_FOUND = "artifact_found"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_STALE = "artifact_is_stale"
    RUNTIME_ERROR = "runtime_error"
    REGISTRY_ERROR = "registry_error"


class Header(str, Enum):
    STATIC_INFERENCE = "static_threshold"
    MODEL_INFERENCE = "model_inference"
    TRAIN_REQUEST = "request_training"


@dataclass
class _BasePayload:
    uuid: str
    config_id: str
    composite_keys: list[str]


payload_t = TypeVar("payload_t", bound=_BasePayload)


@dataclass
class TrainerPayload(_BasePayload):
    metrics: list[str]
    header: Header = Header.TRAIN_REQUEST

    def to_json(self):
        return orjson.dumps(self)


@dataclass(repr=False)
class StreamPayload(_BasePayload):
    data: Matrix
    raw_data: Matrix
    metrics: list[str]
    timestamps: list[int]
    status: Optional[Status] = None
    header: Header = Header.MODEL_INFERENCE
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_data(self, arr: Matrix) -> None:
        self.data = arr

    def get_data(self, original=False) -> npt.NDArray[float]:
        if original:
            return np.ascontiguousarray(self.raw_data, dtype=np.float32)
        return np.ascontiguousarray(self.data, dtype=np.float32)

    def set_status(self, status: Status) -> None:
        self.status = status

    def set_header(self, header: Header) -> None:
        self.header = header

    def set_metadata(self, key: str, value) -> None:
        self.metadata[key] = value

    def get_metadata(self, key: str) -> dict[str, Any]:
        return copy(self.metadata[key])

    def __str__(self) -> str:
        return (
            f'"StreamPayload(header={self.header}, status={self.status}, '
            f'composite_keys={self.composite_keys}, data={list(self.data)})"'
        )

    def __repr__(self) -> str:
        return self.to_json().decode("utf-8")

    def to_json(self) -> bytes:
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY)
