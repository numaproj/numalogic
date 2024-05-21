|        Metric Name        |   Type    |                                     Labels                                     |                     Description                      |
|:-------------------------:|:---------:|:------------------------------------------------------------------------------:|:----------------------------------------------------:|
|      MSG_IN_COUNTER       |  Counter  |                 vertex, composite_key, config_id, pipeline_id                  |                Count msgs flowing in                 |
|   MSG_PROCESSED_COUNTER   |  Counter  |                 vertex, composite_key, config_id, pipeline_id                  |                 Count msgs processed                 |
|      SOURCE_COUNTER       |  Counter  |                 source, composite_key, config_id, pipeline_id                  |   Count artifact source (registry or cache) calls    |
| INSUFFICIENT_DATA_COUNTER |  Counter  |                     composite_key, config_id, pipeline_id                      |        Count insufficient data while Training        |
|   MODEL_STATUS_COUNTER    |  Counter  |             status, vertex, composite_key, config_id, pipeline_id              |              Count status of the model               |
|  DATASHAPE_ERROR_COUNTER  |  Counter  |                     composite_key, config_id, pipeline_id                      |         Count datashape errors in preprocess         |
|    MSG_DROPPED_COUNTER    |  Counter  |                 vertex, composite_key, config_id, pipeline_id                  |                  Count dropped msgs                  |
|    REDIS_ERROR_COUNTER    |  Counter  |                 vertex, composite_key, config_id, pipeline_id                  |                  Count redis errors                  |
|     EXCEPTION_COUNTER     |  Counter  |                 vertex, composite_key, config_id, pipeline_id                  |                   Count exceptions                   |
|   RUNTIME_ERROR_COUNTER   |  Counter  |                 vertex, composite_key, config_id, pipeline_id                  |                 Count runtime errors                 |
|  FETCH_EXCEPTION_COUNTER  |  Counter  |                     composite_key, config_id, pipeline_id                      |    Count exceptions during train data fetch calls    |
|  DATAFRAME_SHAPE_SUMMARY  |  Summary  |                     composite_key, config_id, pipeline_id                      |            len of dataframe for training             |
|        NAN_SUMMARY        |  Summary  |                     composite_key, config_id, pipeline_id                      |              Count nan's in train data               |
|        INF_SUMMARY        |  Summary  |                     composite_key, config_id, pipeline_id                      |              Count inf's in train data               |
|    FETCH_TIME_SUMMARY     |  Summary  |                     composite_key, config_id, pipeline_id                      |                Train Data Fetch time                 |
|        MODEL_INFO         |   Info    |                     composite_key, config_id, pipeline_id                      |                      Model info                      |
|         UDF_TIME          | Histogram |                     composite_key, config_id, pipeline_id                      |          Histogram for udf processing time           |
|    RECORDED_DATA_GAUGE    |   Gauge   | "source", "vertex", "composite_key", "config_id", "pipeline_id", "metric_name" | Gauge metric to observe the mean value of the window |
