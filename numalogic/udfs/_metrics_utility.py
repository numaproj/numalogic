import os
from typing import Final, Optional

from numaprom.monitoring.metrics import (
    PromCounterMetric,
    PromInfoMetric,
    PromSummaryMetric,
    PromGaugeMetric,
)
from numaprom.monitoring.utility import create_metrics_from_config_file

from numalogic import LOGGER
from numalogic._constants import DFAULT_METRICS_CONF_PATH

METRICS_CONFIG_FILE_PATH: Final[str] = os.getenv(
    "DEFAULT_METRICS_CONF_PATH", default=DFAULT_METRICS_CONF_PATH
)


class MetricsSingleton:
    _instance = None
    _metrics = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def load_metrics(self, config_file_path):
        if not self._metrics:
            self._metrics = create_metrics_from_config_file(config_file_path)
        return self._metrics


# helper functions
def _increment_counter(
    counter: PromCounterMetric, labels: Optional[dict], amount: int = 1, is_enabled=True
) -> None:
    """
    Utility function is used to increment the counter.

    Args:
        counter: Counter object
        labels: dict of label keys, value pair
        amount: Amount to increment the counter by
    """
    if is_enabled:
        counter.increment_counter(labels=labels, amount=amount)


def _add_info(info: PromInfoMetric, labels: Optional[dict], data: dict, is_enabled=True) -> None:
    """
    Utility function is used to add the info.

    Args:
        info: Info object
        labels: dict of label keys, value pair
        data: Dictionary of data
    """
    if is_enabled:
        info.add_info(labels=labels, data=data)


def _add_summary(
    summary: PromSummaryMetric, labels: Optional[dict], data: float, is_enabled=True
) -> None:
    """
    Utility function is used to add the summary.

    Args:
        summary: Summary object
        labels: dict of labels key, value pair
        data: Summary value
    """
    if is_enabled:
        summary.add_observation(labels=labels, value=data)


def _set_gauge(
    gauge: PromGaugeMetric, labels: Optional[dict], data: float, is_enabled=True
) -> None:
    """
    Utility function is used to add the info.
    Args:
        gauge: Gauge object
        labels: dict of label keys, value pair
        data: data.
    """
    if is_enabled:
        gauge.set_gauge(labels=labels, data=data)


LOGGER.info("Loading metrics from config file: %s", METRICS_CONFIG_FILE_PATH)
_METRICS = MetricsSingleton().load_metrics(METRICS_CONFIG_FILE_PATH)
