from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union, TypeVar

import numpy as np
import numpy.typing as npt
import orjson
import pandas as pd

Vector = list[float]
Matrix = Union[Vector, list[Vector], npt.NDArray[float]]


class Status(str, Enum):
    """Status is the enum that is used to identify the status of the payload."""

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
    """Header is the enum that is used to identify the type of payload."""

    STATIC_INFERENCE = "static_threshold"
    MODEL_INFERENCE = "model_inference"
    TRAIN_REQUEST = "request_training"
    MODEL_STALE = "model_stale"


@dataclass
class _BasePayload:
    """_BasePayload is the base data structure that is passed."""

    uuid: str
    composite_keys: list[str]


PayloadType = TypeVar("PayloadType", bound=_BasePayload)


@dataclass
class TrainerPayload(_BasePayload):
    """
    TrainerPayload is the data structure that is passed
    around in the system when a training request is made.
    """

    metric: str
    header: Header = Header.TRAIN_REQUEST

    def to_json(self):
        return orjson.dumps(self)


@dataclass(repr=False)
class StreamPayload(_BasePayload):
    """StreamPayload is the main data structure that is passed around in the system."""

    data: Matrix
    raw_data: Matrix
    metrics: list[str]
    timestamps: list[int]
    status: dict[str, Status] = field(default_factory=dict)
    header: dict[str, Header] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_df(self, original=False) -> pd.DataFrame:
        return pd.DataFrame(self.get_data(original), columns=self.metrics)

    def set_data(self, arr: Matrix) -> None:
        self.data = arr

    def set_metric_data(self, metric: str, arr: Matrix) -> None:
        _df = self.get_df().copy()
        _df[metric] = arr
        self.set_data(np.asarray(_df.values.tolist()))

    def get_metric_arr(self, metric: str) -> npt.NDArray[float]:
        return self.get_df()[metric].values

    def get_data(self, original=False) -> npt.NDArray[float]:
        if original:
            return np.asarray(self.raw_data)
        return np.asarray(self.data)

    def set_status(self, metric: str, status: Status) -> None:
        self.status[metric] = status

    def set_header(self, metric: str, header: Header) -> None:
        self.header[metric] = header

    def set_metric_metadata(self, metric: str, key: str, value) -> None:
        if metric in self.metadata.keys():
            self.metadata[metric][key] = value
        else:
            self.metadata[metric] = {key: value}

    def set_metadata(self, key: str, value) -> None:
        self.metadata[key] = value

    def get_metadata(self, key: str) -> dict[str, Any]:
        return copy(self.metadata[key])

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return "header: {}, status: {}, composite_keys: {}, data: {}, metadata: {}}}".format(
            self.header,
            self.status,
            self.composite_keys,
            list(self.data),
            self.metadata,
        )

    def to_json(self):
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY)


@dataclass
class InputPayload:
    """Input payload."""

    start_time: int
    end_time: int
    data: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_json(self):
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY)


@dataclass
class OutputPayload:
    """Output payload."""

    timestamp: int
    unified_anomaly: float
    data: dict[str, Any]
    metadata: dict[str, Any]

    def to_json(self):
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY)
