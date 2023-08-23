import datetime
import json

import numpy as np
from pynumaflow.function import Datum, DatumMetadata
from sklearn.pipeline import make_pipeline

from numalogic.config import PreprocessFactory


def input_json_from_file(data_path: str) -> Datum:
    """
    Read input json from file and return Datum object
    Args:
        data_path: file path.

    Returns
    -------
    Datum object

    """
    with open(data_path) as fp:
        data = json.load(fp)
    if not isinstance(data, bytes):
        data = json.dumps(data).encode("utf-8")

    return Datum(
        keys=["service-mesh", "1", "2"],
        value=data,
        event_time=datetime.datetime.now(),
        watermark=datetime.datetime.now(),
        metadata=DatumMetadata(msg_id="", num_delivered=0),
    )


def store_in_redis(stream_conf, registry):
    """Store preprocess artifacts in redis."""
    preproc_clfs = []
    preproc_factory = PreprocessFactory()
    for _cfg in stream_conf.numalogic_conf.preprocess:
        _clf = preproc_factory.get_instance(_cfg)
        preproc_clfs.append(_clf)
    if any([_conf.stateful for _conf in stream_conf.numalogic_conf.preprocess]):
        preproc_clf = make_pipeline(*preproc_clfs)
        preproc_clf.fit(np.asarray([[1, 3], [4, 6]]))
        registry.save(
            skeys=stream_conf.composite_keys,
            dkeys=[_conf.name for _conf in stream_conf.numalogic_conf.preprocess],
            artifact=preproc_clf,
        )
