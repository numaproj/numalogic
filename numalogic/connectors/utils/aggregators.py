def quantiles_doubles_sketch(
        raw_column: str, aggregator_name: str, k: int = 128, max_stream_length: int = 1000000000
) -> dict:
    """

    Args:
        raw_column: name of the column in druid
        aggregator_name: arbitrary aggregator name
        k: controls accuracy, higher the better.  Must be a power of 2 from 2 to 32768
        max_stream_length: this parameter defines the number of items that can be presented to
        each sketch before it may need to move from off-heap to on-heap memory.

    Returns: quantilesDoublesSketch aggregator dict

    """
    return {
        "type": "quantilesDoublesSketch",
        "name": aggregator_name,
        "fieldName": raw_column,
        "k": k,
        "maxStreamLength": max_stream_length,
    }
