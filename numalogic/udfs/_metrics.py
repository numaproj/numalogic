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
    "UDF_TIME",
    "FETCH_TIME_SUMMARY",
    "DATAFRAME_SHAPE_SUMMARY",
    "FETCH_EXCEPTION_COUNTER",
    "INF_SUMMARY",
]

SOURCE_COUNTER = PromCounterMetric(
    "numalogic_artifact_source_counter",
    "Count artifact source calls",
    ["source", "composite_key", "config_id"],
)

# Trainer Counters
INSUFFICIENT_DATA_COUNTER = PromCounterMetric(
    "numalogic_insufficient_data_counter",
    "Count insufficient data while Training",
    ["composite_key", "config_id"],
)
MODEL_STATUS_COUNTER = PromCounterMetric(
    "numalogic_new_model_counter",
    "Count status of the model",
    ["status", "vertex", "composite_key", "config_id"],
)

# Data Counters
DATASHAPE_ERROR_COUNTER = PromCounterMetric(
    "numalogic_datashape_error_counter",
    "Count datashape errors in preprocess",
    ["composite_key", "config_id"],
)
MSG_DROPPED_COUNTER = PromCounterMetric(
    "numalogic_msg_dropped_counter", "Count dropped msgs", ["vertex", "composite_key", "config_id"]
)

# ERROR Counters
REDIS_ERROR_COUNTER = PromCounterMetric(
    "numalogic_redis_error_counter", "Count redis errors", ["vertex", "composite_key", "config_id"]
)
EXCEPTION_COUNTER = PromCounterMetric(
    "numalogic_exception_counter", "Count exceptions", ["vertex", "composite_key", "config_id"]
)
RUNTIME_ERROR_COUNTER = PromCounterMetric(
    "numalogic_runtime_error_counter",
    "Count runtime errors",
    ["vertex", "composite_key", "config_id"],
)

# TRAIN COUNTERS
FETCH_EXCEPTION_COUNTER = PromCounterMetric(
    "numalogic_fetch_exception_counter",
    "count exceptions during fetch call",
    ["composite_key", "config_id"],
)

# TRAIN SUMMARY
DATAFRAME_SHAPE_SUMMARY = PromSummaryMetric(
    "numalogic_dataframe_shape_summary",
    "len of dataframe for training",
    ["composite_key", "config_id"],
)
NAN_SUMMARY = PromSummaryMetric(
    "numalogic_nan_counter", "Count nan's in train data", ["composite_key", "config_id"]
)
INF_SUMMARY = PromSummaryMetric(
    "numalogic_inf_counter", "Count inf's in train data", ["composite_key", "config_id"]
)
FETCH_TIME_SUMMARY = PromSummaryMetric(
    "numalogic_fetch_time_summary", "Train data fetch time", ["composite_key", "config_id"]
)

MSG_IN_COUNTER = PromCounterMetric(
    "numalogic_msg_in_counter", "Count msgs flowing in", ["vertex", "composite_key", "config_id"]
)
MSG_PROCESSED_COUNTER = PromCounterMetric(
    "numalogic_msg_processed_counter",
    "Count msgs processed",
    ["vertex", "composite_key", "config_id"],
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

UDF_TIME = Histogram(
    "numalogic_udf_time_histogram",
    "Histogram for udf processing time",
    buckets=buckets,
)
