from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union, TypeVar, Optional

import numpy as np
import numpy.typing as npt
import orjson

from numalogic._constants import NUMALOGIC_METRICS

Vector = list[float]
Matrix = Union[Vector, list[Vector], npt.ArrayLike]


class Status(str, Enum):
    """Status of the current payload."""

    NONE = "none"
    ARTIFACT_FOUND = "artifact_found"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_STALE = "artifact_is_stale"
    RUNTIME_ERROR = "runtime_error"
    REGISTRY_ERROR = "registry_error"


class Header(str, Enum):
    """Payload header indicating the path to take."""

    STATIC_INFERENCE = "static_threshold"
    MODEL_INFERENCE = "model_inference"
    TRAIN_REQUEST = "request_training"


@dataclass
class _BasePayload:
    uuid: str
    config_id: str
    pipeline_id: str
    composite_keys: list[str]


payload_t = TypeVar("payload_t", bound=_BasePayload)


@dataclass
class TrainerPayload(_BasePayload):
    """Payload for training request."""

    metrics: list[str]
    header: Header = Header.TRAIN_REQUEST

    def to_json(self):
        return orjson.dumps(self)


@dataclass(repr=False)
class StreamPayload(_BasePayload):
    """Payload that gets propagated along the numalogic UDFs."""

    data: Matrix
    raw_data: Matrix
    metrics: list[str]
    timestamps: list[int]
    status: Status = Status.NONE
    header: Header = Header.MODEL_INFERENCE
    artifact_versions: dict[str, dict] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            _ = self.metadata["numalogic_opex_tags"]["source"]
        except KeyError:
            self.metadata["numalogic_opex_tags"] = {"source": NUMALOGIC_METRICS}

    @property
    def start_ts(self) -> int:
        return int(self.timestamps[0])

    @property
    def end_ts(self) -> int:
        return int(self.timestamps[-1])

    def set_data(self, arr: Matrix) -> None:
        self.data = arr

    def get_data(self, original=False, metrics: Optional[list[str]] = None) -> npt.NDArray[float]:
        if original:
            arr = np.ascontiguousarray(self.raw_data, dtype=np.float32)
        else:
            arr = np.ascontiguousarray(self.data, dtype=np.float32)
        if metrics:
            return arr[:, [self.metrics.index(m) for m in metrics]]
        return arr

    def get_metadata(self, key: str) -> dict[str, Any]:
        return copy(self.metadata[key])

    def __str__(self) -> str:
        return (
            f'"StreamPayload(header={self.header}, status={self.status}, '
            f'composite_keys={self.composite_keys}, data={list(self.data)})"'
            f"artifact_versions={self.artifact_versions}"
            f"metadata={self.metadata}"
        )

    def __repr__(self) -> str:
        return self.to_json().decode("utf-8")

    def to_json(self) -> bytes:
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY)


@dataclass
class OutputPayload(_BasePayload):
    """Payload for output data from the numalogic pipeline."""

    timestamp: int
    unified_anomaly: float  # TODO: change to a more generic name
    data: dict[str, Any]
    metadata: dict[str, Any]

    def to_json(self):
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY)
