from prometheus_client import Histogram
from numalogic.monitoring.metrics import PromCounterMetric, PromInfoMetric, PromSummaryMetric

__all__ = [
    "SOURCE_COUNTER",
    "INSUFFICIENT_DATA_COUNTER",
    "MODEL_STATUS_COUNTER",
    "NAN_SUMMARY",
    "DATASHAPE_ERROR_COUNTER",
    "MSG_DROPPED_COUNTER",
    "REDIS_ERROR_COUNTER",
    "EXCEPTION_COUNTER",
    "RUNTIME_ERROR_COUNTER",
    "MSG_IN_COUNTER",
    "MSG_PROCESSED_COUNTER",
    "MODEL_INFO",
    "INFER_TIME",
    "PREPROC_TIME",
    "POSTPROC_TIME",
    "DATAFRAME_SHAPE_SUMMARY",
    "FETCH_TIME",
    "FETCH_EXCEPTION_COUNTER",
    "TRAIN_TIME",
    "INF_SUMMARY",
]

SOURCE_COUNTER = PromCounterMetric(
    "numalogic_artifact_source_count",
    "Count artifact source calls",
    ["source", "composite_key", "config_id"],
)

# Trainer Counters
INSUFFICIENT_DATA_COUNTER = PromCounterMetric(
    "numalogic_insufficient_data_count", "Count insufficient data", ["composite_key", "config_id"]
)
MODEL_STATUS_COUNTER = PromCounterMetric(
    "numalogic_new_model_count",
    "Count new models",
    ["status", "vertex", "composite_key", "config_id"],
)

# Data Counters
DATASHAPE_ERROR_COUNTER = PromCounterMetric(
    "numalogic_datashape_error_count", "Count datashape errors", ["composite_key", "config_id"]
)
MSG_DROPPED_COUNTER = PromCounterMetric(
    "numalogic_msg_dropped_count", "Count dropped msgs", ["vertex", "composite_key", "config_id"]
)

# ERROR Counters
REDIS_ERROR_COUNTER = PromCounterMetric(
    "numalogic_redis_error_count", "Count redis errors", ["vertex", "composite_key", "config_id"]
)
EXCEPTION_COUNTER = PromCounterMetric(
    "numalogic_exception_count", "Count exceptions", ["vertex", "composite_key", "config_id"]
)
RUNTIME_ERROR_COUNTER = PromCounterMetric(
    "numalogic_runtime_error_count",
    "Count runtime errors",
    ["vertex", "composite_key", "config_id"],
)

# TRAIN COUNTERS
FETCH_EXCEPTION_COUNTER = PromCounterMetric(
    "numalogic_fetch_exception_count",
    "count exceptions during fetch call",
    ["composite_key", "config_id"],
)

# TRAIN SUMMARY
DATAFRAME_SHAPE_SUMMARY = PromSummaryMetric(
    "numalogic_dataframe_shape", "shape of dataframe", ["composite_key", "config_id"]
)
NAN_SUMMARY = PromSummaryMetric(
    "numalogic_nan_count", "Count nan's in data", ["composite_key", "config_id"]
)
INF_SUMMARY = PromSummaryMetric(
    "numalogic_inf_count", "Count inf's in data", ["composite_key", "config_id"]
)

MSG_IN_COUNTER = PromCounterMetric(
    "numalogic_msg_in_counter", "Count msgs in", ["vertex", "composite_key", "config_id"]
)
MSG_PROCESSED_COUNTER = PromCounterMetric(
    "numalogic_msg_processed_counter", "Count msgs ", ["vertex", "composite_key", "config_id"]
)

# Info
MODEL_INFO = PromInfoMetric("numalogic_model_info", "Model info", ["composite_key", "config_id"])

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

INFER_TIME = Histogram(
    "numalogic_histogram_infer",
    "Histogram",
    buckets=buckets,
)
PREPROC_TIME = Histogram(
    "numalogic_histogram_preproc",
    "Histogram",
    buckets=buckets,
)
POSTPROC_TIME = Histogram(
    "numalogic_histogram_postproc",
    "Histogram",
    buckets=buckets,
)
TRAIN_TIME = Histogram(
    "numalogic_histogram_train",
    "Histogram",
    buckets=buckets,
)

FETCH_TIME = Histogram(
    "numalogic_histogram_train_fetch",
    "Histogram",
    buckets=buckets,
)
