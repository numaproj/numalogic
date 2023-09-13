import datetime
import json
from collections import namedtuple

import numpy as np
from pynumaflow.function import Datum, DatumMetadata
from sklearn.pipeline import make_pipeline

from numalogic.config import PreprocessFactory
from numalogic.models.autoencoder.variants import VanillaAE


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


def store_in_redis(pl_conf, registry):
    """Store preprocess artifacts in redis."""
    preproc_clfs = []
    preproc_factory = PreprocessFactory()
    for _cfg in pl_conf.stream_confs["druid-config"].numalogic_conf.preprocess:
        _clf = preproc_factory.get_instance(_cfg)
        preproc_clfs.append(_clf)
    if any(
        [_conf.stateful for _conf in pl_conf.stream_confs["druid-config"].numalogic_conf.preprocess]
    ):
        artifact_tuple = namedtuple("artifact_tuple", ["dkeys", "artifact"])
        preproc_clf = make_pipeline(*preproc_clfs)
        preproc_clf.fit(np.asarray([[1, 3], [4, 6]]))
        registry.save_multiple(
            skeys=pl_conf.stream_confs["druid-config"].composite_keys,
            dict_artifacts={
                "inference": artifact_tuple(dkeys=["AE"], artifact=VanillaAE(10)),
                "preproc": artifact_tuple(
                    dkeys=[
                        _conf.name
                        for _conf in pl_conf.stream_confs["druid-config"].numalogic_conf.preprocess
                    ],
                    artifact=preproc_clf,
                ),
            },
        )
