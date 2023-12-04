import logging
from typing import Optional

from prometheus_client import Counter, Info, Summary

_LOGGER = logging.getLogger(__name__)


class PromCounterMetric:
    """
    Class is used to create a counter object and increment it.

    Args:
        name: Name of the metric
        description: Description of the metric
        labels: List of labels
    """

    __slots__ = ("name", "description", "labels", "counter")

    def __init__(self, name: str, description: str, labels: list[str]) -> None:
        self.name = name
        self.description = description
        self.labels = labels
        self.counter = Counter(name, description, labels)

    def increment_counter(self, *args, amount: float = 1) -> None:
        """
        Utility function is used to increment the counter.

        Args:
            *args: List of labels
            amount: Amount to increment the counter by
        """
        self.counter.labels(*args).inc(amount=amount)


class PromInfoMetric:
    """
    Class is used to create an info object and increment it.

    Args:
        name: Name of the metric
        description: Description of the metric
        labels: List of labels
    """

    __slots__ = ("name", "description", "labels", "info")

    def __init__(self, name: str, description: str, labels: Optional[list[str]]) -> None:
        self.name = name
        self.description = description
        self.labels = labels
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
        self.info.labels(*args).info(data)


class PromSummaryMetric:
    """
    Class is used to create a histogram object and increment it.

    Args:
        name: Name of the metric
        description: Description of the metric
        labels: List of labels
    """

    __slots__ = ("name", "description", "labels", "summary")

    def __init__(self, name: str, description: str, labels: Optional[list[str]]) -> None:
        self.name = name
        self.description = description
        self.labels = labels
        self.summary = Summary(name, description, labels)

    def add_observation(self, *args, value: float) -> None:
        """
        Utility function is to update the summary value with the given value.

        Args:
            *args: List of labels
            value: Value to be updated
        """
        self.summary.labels(*args).observe(amount=value)
