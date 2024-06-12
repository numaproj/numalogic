from typing import Optional, Any

from numaprom.monitoring.utility import get_metric
from omegaconf import OmegaConf


def create_metrics_from_config_file(config_file_path: str) -> dict[str, Any]:
    config = OmegaConf.load(config_file_path)
    metrics = {}
    for metric_config in config.get("numalogic_metrics", []):
        metric_type = metric_config["type"]
        for metric in metric_config["metrics"]:
            name = metric["name"]
            description = metric.get("description", "")
            label_pairs = metric.get("label_pairs", {})
            static_label_pairs = metric.get("static_label_pairs", {})
            metrics[name] = get_metric(
                metric_type, name, description, label_pairs, static_label_pairs
            )
    return metrics


class MetricsSingleton:
    _instance = None
    _metrics = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def load_metrics(self, config_file_path: str):
        if not self._metrics:
            if config_file_path is None:
                raise ValueError("file path is required to load metrics")
            self._metrics = create_metrics_from_config_file(config_file_path)

    def get_metrics(self) -> dict:
        return self._metrics


# helper functions
def _increment_counter(
    counter: str, labels: Optional[dict], amount: int = 1, is_enabled=True
) -> None:
    """
    Utility function is used to increment the counter.

    Args:
        counter: Counter object
        labels: dict of label keys, value pair
        amount: Amount to increment the counter by
    """
    _metrics = MetricsSingleton().get_metrics()
    if is_enabled and counter in _metrics:
        _metrics[counter].increment_counter(labels=labels, amount=amount)


def _add_info(info: str, labels: Optional[dict], data: dict, is_enabled=True) -> None:
    """
    Utility function is used to add the info.

    Args:
        info: Info object
        labels: dict of label keys, value pair
        data: Dictionary of data
    """
    _metrics = MetricsSingleton().get_metrics()
    if is_enabled and info in _metrics:
        _metrics[info].add_info(labels=labels, data=data)


def _add_summary(summary: str, labels: Optional[dict], data: float, is_enabled=True) -> None:
    """
    Utility function is used to add the summary.

    Args:
        summary: Summary object
        labels: dict of labels key, value pair
        data: Summary value
    """
    _metrics = MetricsSingleton().get_metrics()
    if is_enabled and summary in _metrics:
        _metrics[summary].add_observation(labels=labels, value=data)


def _set_gauge(gauge: str, labels: Optional[dict], data: float, is_enabled=True) -> None:
    """
    Utility function is used to add the info.
    Args:
        gauge: Gauge object
        labels: dict of label keys, value pair
        data: data.
    """
    _metrics = MetricsSingleton().get_metrics()
    if is_enabled and gauge in _metrics:
        _metrics[gauge].set_gauge(labels=labels, data=data)
