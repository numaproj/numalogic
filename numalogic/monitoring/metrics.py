import logging
from typing import Optional

from prometheus_client import Counter, Info, Summary

_LOGGER = logging.getLogger(__name__)


class BaseMetricClass:
    """
    Base class for metrics.

    Args:
        name: Name of the metric
        description: Description of the metric
        labels: List of labels
    """

    __slots__ = ("name", "description", "labels")

    def __init__(self, name: str, description: str, labels: Optional[list[str]]) -> None:
        self.name = name
        self.description = description
        self.labels = labels


class PromCounterMetric(BaseMetricClass):
    """Class is used to create a counter object and increment it."""

    __slots__ = "counter"

    def __init__(self, name: str, description: str, labels: list[str]) -> None:
        super().__init__(name, description, labels)
        self.counter = Counter(name, description, labels)

    def increment_counter(self, *args, amount: int = 1) -> None:
        """
        Utility function is used to increment the counter.

        Args:
            *args: List of labels
            amount: Amount to increment the counter by
        """
        if len(args) != len(self.labels):
            raise ValueError(f"Labels mismatch with the definition: {self.labels}")
        self.counter.labels(*args).inc(amount=amount)


class PromInfoMetric(BaseMetricClass):
    """Class is used to create an info object and increment it."""

    __slots__ = ("name", "description", "labels", "info")

    def __init__(self, name: str, description: str, labels: Optional[list[str]]) -> None:
        super().__init__(name, description, labels)
        self.info = Info(name, description, labels)

    def add_info(
        self,
        *args,
        data: dict,
    ) -> None:
        """
        Utility function is used to increment the info.

        Args:
            *args: List of labels
            data: Dictionary of data
        """
        if len(args) != len(self.labels):
            raise ValueError(f"Labels mismatch with the definition: {self.labels}")
        self.info.labels(*args).info(data)


class PromSummaryMetric(BaseMetricClass):
    """Class is used to create a histogram object and increment it."""

    __slots__ = ("name", "description", "labels", "summary")

    def __init__(self, name: str, description: str, labels: Optional[list[str]]) -> None:
        super().__init__(name, description, labels)
        self.summary = Summary(name, description, labels)

    def add_observation(self, *args, value: float) -> None:
        """
        Utility function is to update the summary value with the given value.

        Args:
            *args: List of labels
            value: Value to be updated
        """
        if len(args) != len(self.labels):
            raise ValueError(f"Labels mismatch with the definition: {self.labels}")
        self.summary.labels(*args).observe(amount=value)
