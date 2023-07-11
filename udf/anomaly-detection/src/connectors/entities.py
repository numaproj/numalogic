from dataclasses import dataclass
from typing import Optional, Dict, Union, Self

from orjson import orjson


@dataclass(repr=False)
class PrometheusPayload:
    timestamp_ms: int
    name: str
    namespace: str
    subsystem: Optional[str]
    type: str
    value: float
    labels: Dict[str, str]

    def as_json(self) -> bytes:
        return orjson.dumps(
            {
                "TimestampMs": self.timestamp_ms,
                "Name": self.name,
                "Namespace": self.namespace,
                "Subsystem": self.subsystem,
                "Type": self.type,
                "Value": self.value,
                "Labels": self.labels,
            }
        )

    @classmethod
    def from_json(cls, json_obj: Union[bytes, str]) -> Self:
        obj = orjson.loads(json_obj)
        return cls(
            timestamp_ms=obj["TimestampMs"],
            name=obj["Name"],
            namespace=obj["Namespace"],
            subsystem=obj["Subsystem"],
            type=obj["Type"],
            value=obj["Value"],
            labels=obj["Labels"],
        )

    def __repr__(self) -> str:
        return (
            "{timestamp_ms: %s, name: %s, namespace: %s, "
            "subsystem: %s, type: %s, value: %s, labels: %s}"
            % (
                self.timestamp_ms,
                self.name,
                self.namespace,
                self.subsystem,
                self.type,
                self.value,
                self.labels,
            )
        )
