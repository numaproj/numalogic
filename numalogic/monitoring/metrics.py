import logging
from typing import Optional

from prometheus_client import Counter, Info

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

    def increment_counter(self, *args) -> None:
        """
        Utility function is used to increment the counter.

        Args:
            *args: List of labels
        """
        self.counter.labels(*args).inc()

    def get_counter(self):
        """Utility function is used to get the counter."""
        return self.counter


class PromInfoMetric:
    """
    Class is used to create an info object and increment it.

    Args:
        name: Name of the metric
        description: Description of the metric
        labels: List of labels
    """

    def __init__(self, name: str, description: str, labels: Optional[list[str]]) -> None:
        self.name = name
        self.description = description
        self.labels = labels
        self.info = Info(name, description, labels)

    def add_info(self, *args, data: dict) -> None:
        """
        Utility function is used to increment the info.

        Args:
            *args: List of labels
            data: Dictionary of data
        """
        self.info.labels(*args).info(data)
