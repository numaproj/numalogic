from collections.abc import Sequence

from prometheus_client import Histogram

from numalogic.monitoring.metrics import (
    PromCounterMetric,
    PromInfoMetric,
    PromSummaryMetric,
    PromGaugeMetric,
)

# Define metrics

# COUNTERS
SOURCE_COUNTER = PromCounterMetric(
    "numalogic_artifact_source_counter",
    "Count artifact source calls",
    ["artifact_source", "source", "vertex", "composite_key", "config_id", "pipeline_id"],
)

INSUFFICIENT_DATA_COUNTER = PromCounterMetric(
    "numalogic_insufficient_data_counter",
    "Count insufficient data while Training",
    ["source", "composite_key", "config_id", "pipeline_id"],
)
MODEL_STATUS_COUNTER = PromCounterMetric(
    "numalogic_new_model_counter",
    "Count status of the model",
    ["source", "status", "vertex", "composite_key", "config_id", "pipeline_id"],
)

DATASHAPE_ERROR_COUNTER = PromCounterMetric(
    "numalogic_datashape_error_counter",
    "Count datashape errors in preprocess",
    ["source", "composite_key", "config_id", "pipeline_id"],
)
MSG_DROPPED_COUNTER = PromCounterMetric(
    "numalogic_msg_dropped_counter",
    "Count dropped msgs",
    ["source", "vertex", "composite_key", "config_id", "pipeline_id"],
)

REDIS_ERROR_COUNTER = PromCounterMetric(
    "numalogic_redis_error_counter",
    "Count redis errors",
    ["source", "vertex", "composite_key", "config_id", "pipeline_id"],
)
EXCEPTION_COUNTER = PromCounterMetric(
    "numalogic_exception_counter",
    "Count exceptions",
    ["source", "vertex", "composite_key", "config_id", "pipeline_id"],
)
RUNTIME_ERROR_COUNTER = PromCounterMetric(
    "numalogic_runtime_error_counter",
    "Count runtime errors",
    ["source", "vertex", "composite_key", "config_id", "pipeline_id"],
)

FETCH_EXCEPTION_COUNTER = PromCounterMetric(
    "numalogic_fetch_exception_counter",
    "count exceptions during fetch call",
    ["source", "composite_key", "config_id", "pipeline_id"],
)

MSG_IN_COUNTER = PromCounterMetric(
    "numalogic_msg_in_counter",
    "Count msgs flowing in",
    ["source", "vertex", "composite_key", "config_id", "pipeline_id"],
)
MSG_PROCESSED_COUNTER = PromCounterMetric(
    "numalogic_msg_processed_counter",
    "Count msgs processed",
    ["source", "vertex", "composite_key", "config_id", "pipeline_id"],
)

# SUMMARY
DATAFRAME_SHAPE_SUMMARY = PromSummaryMetric(
    "numalogic_dataframe_shape_summary",
    "len of dataframe for training",
    ["source", "composite_key", "config_id", "pipeline_id"],
)
NAN_SUMMARY = PromSummaryMetric(
    "numalogic_nan_summary",
    "Count nan's in train data",
    ["source", "composite_key", "config_id", "pipeline_id"],
)
INF_SUMMARY = PromSummaryMetric(
    "numalogic_inf_summary",
    "Count inf's in train data",
    ["source", "composite_key", "config_id", "pipeline_id"],
)
FETCH_TIME_SUMMARY = PromSummaryMetric(
    "numalogic_fetch_time_summary",
    "Train data fetch time",
    ["source", "composite_key", "config_id", "pipeline_id"],
)
# Gauge Metrics
RECORDED_DATA_GAUGE = PromGaugeMetric(
    "numalogic_recorded_value_gauge",
    "Gauge metric to observe the mean value of the window",
    ["source", "vertex", "composite_key", "config_id", "pipeline_id", "metric_name"],
)

# Info
MODEL_INFO = PromInfoMetric(
    "numalogic_model_info", "Model info", ["source", "composite_key", "config_id", "pipeline_id"]
)

# HISTOGRAM
buckets = (
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
)

UDF_TIME = Histogram(
    "numalogic_udf_time_histogram",
    "Histogram for udf processing time",
    buckets=buckets,
)


# helper functions


def _increment_counter(counter: PromCounterMetric, labels: Sequence[str], amount: int = 1) -> None:
    """
    Utility function is used to increment the counter.

    Args:
        counter: Counter object
        labels: Sequence of labels
        amount: Amount to increment the counter by
    """
    counter.increment_counter(*labels, amount=amount)


def _add_info(info: PromInfoMetric, labels: Sequence[str], data: dict) -> None:
    """
    Utility function is used to add the info.

    Args:
        info: Info object
        labels: Sequence of labels
        data: Dictionary of data
    """
    info.add_info(*labels, data=data)


def _add_summary(summary: PromSummaryMetric, labels: Sequence[str], data: float) -> None:
    """
    Utility function is used to add the summary.

    Args:
        summary: Summary object
        labels: Sequence of labels
        data: Summary value
    """
    summary.add_observation(*labels, value=data)


def _set_gauge(gauge: PromGaugeMetric, labels: Sequence[str], data: float) -> None:
    """
    Utility function is used to add the info.
    Args:
        gauge: Gauge object
        labels: Sequence of labels
        data: data.
    """
    gauge.set_gauge(*labels, data=data)
