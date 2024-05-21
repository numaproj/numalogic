import logging
from typing import Optional
from collections.abc import Sequence

from prometheus_client import Counter, Info, Summary, Gauge

_LOGGER = logging.getLogger(__name__)


class _BaseMetric:
    __slots__ = ("name", "description", "label_keys")

    """
    Base class for metrics.

    Args:
        name: Name of the metric
        description: Description of the metric
        label_keys: List of labels
    """

    def __init__(self, name: str, description: str, label_keys: Optional[list[str]]) -> None:
        self.name = name
        self.description = description
        self.label_keys = label_keys


class PromCounterMetric(_BaseMetric):
    """Class is used to create a counter object and increment it."""

    __slots__ = "counter"

    def __init__(self, name: str, description: str, label_keys: list[str]) -> None:
        super().__init__(name, description, label_keys)
        self.counter = Counter(name, description, label_keys)

    def increment_counter(self, *label_values: Sequence[str], amount: int = 1) -> None:
        """
        Utility function is used to increment the counter.

        Args:
            *label_values: List of labels
            amount: Amount to increment the counter by
        """
        if len(label_values) != len(self.label_keys):
            raise ValueError(f"Labels length mismatch with the definition: {self.label_keys}")
        self.counter.labels(*label_values).inc(amount=amount)


class PromInfoMetric(_BaseMetric):
    """Class is used to create an info object and increment it."""

    __slots__ = "info"

    def __init__(self, name: str, description: str, label_keys: Optional[list[str]]) -> None:
        super().__init__(name, description, label_keys)
        self.info = Info(name, description, label_keys)

    def add_info(
        self,
        *label_values: Sequence[str],
        data: dict,
    ) -> None:
        """
        Utility function is used to increment the info.

        Args:
            *label_values: List of labels
            data: Dictionary of data
        """
        if len(label_values) != len(self.label_keys):
            raise ValueError(f"Labels length mismatch with the definition: {self.label_keys}")
        self.info.labels(*label_values).info(data)


class PromSummaryMetric(_BaseMetric):
    """Class is used to create a histogram object and increment it."""

    __slots__ = "summary"

    def __init__(self, name: str, description: str, label_keys: Optional[list[str]]) -> None:
        super().__init__(name, description, label_keys)
        self.summary = Summary(name, description, label_keys)

    def add_observation(self, *label_values, value: float) -> None:
        """
        Utility function is to update the summary value with the given value.

        Args:
            *label_values: List of labels
            value: Value to be updated
        """
        if len(label_values) != len(self.label_keys):
            raise ValueError(f"Labels length mismatch with the definition: {self.label_keys}")
        self.summary.labels(*label_values).observe(amount=value)


class PromGaugeMetric(_BaseMetric):
    """Class is used to create an info object and increment it."""

    __slots__ = "info"

    def __init__(self, name: str, description: str, label_keys: Optional[list[str]]) -> None:
        super().__init__(name, description, label_keys)
        self.info = Gauge(name, description, label_keys)

    def set_gauge(
        self,
        *label_values: Sequence[str],
        data: float,
    ) -> None:
        """
        Utility function is used to increment the info.
        Args:
            *label_values: List of labels
            data: float data.
        """
        if len(label_values) != len(self.label_keys):
            raise ValueError(f"Labels mismatch with the definition: {self.label_keys}")
        self.info.labels(*label_values).set(data)
