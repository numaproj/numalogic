import datetime
import json
import os
import sys
import fakeredis
import numpy as np
import pandas as pd
from typing import Union, Optional
from unittest import mock
from unittest.mock import MagicMock, patch, Mock
from sklearn.preprocessing import MinMaxScaler

from numalogic.models.autoencoder.variants import VanillaAE, LSTMAE, SparseVanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.registry import ArtifactData, RedisRegistry
from pynumaflow.function import Datum
from pynumaflow.function._dtypes import DROP, DatumMetadata

from src._constants import TESTS_DIR, POSTPROC_VTX_KEY
from src.udf import Preprocess, Inference, Threshold
from src.watcher import ConfigManager
from tests import mock_configs

sys.modules["numaprom.mlflow"] = MagicMock()
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")


def mockenv(**envvars):
    return mock.patch.dict(os.environ, envvars, clear=True)


def get_datum(keys: list[str], data: str or bytes) -> Datum:
    if type(data) is not bytes:
        data = json.dumps(data).encode("utf-8")

    if not keys:
        keys = ["random_key"]

    return Datum(
        keys=keys,
        value=data,
        event_time=datetime.datetime.now(),
        watermark=datetime.datetime.now(),
        metadata=DatumMetadata(msg_id="", num_delivered=0),
    )


def get_stream_data(data_path: str) -> dict[str, Union[dict, str, list]]:
    with open(data_path) as fp:
        data = json.load(fp)
    return data


def get_mock_redis_client():
    server = fakeredis.FakeServer()
    redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
    return redis_client


def get_prepoc_input(keys: list[str], data_path: str) -> Datum:
    data = get_stream_data(data_path)
    return get_datum(keys, data)


@patch.object(ConfigManager, "load_configs", Mock(return_value=mock_configs()))
def get_inference_input(keys: list[str], data_path: str, prev_clf_exists=True) -> Optional[Datum]:
    preproc_input = get_prepoc_input(keys, data_path)
    _mock_return = return_preproc_clf(2) if prev_clf_exists else None
    with patch.object(RedisRegistry, "load", Mock(return_value=_mock_return)):
        msg = Preprocess().run(keys, preproc_input)[0]

        if len(msg.tags) > 0 and msg.tags[0] == DROP:
            if not msg.tags[0] == DROP:
                return None
    return get_datum(keys, msg.value)


def get_threshold_input(
    keys: list[str], data_path: str, prev_clf_exists=True, prev_model_stale=False
) -> Optional[Datum]:
    inference_input = get_inference_input(keys, data_path)
    if prev_model_stale:
        _mock_return = return_stale_model()
    elif prev_clf_exists:
        _mock_return = return_mock_model_state_dict()
    else:
        _mock_return = None
    with patch.object(RedisRegistry, "load", Mock(return_value=_mock_return)):
        msg = Inference().run(keys, inference_input)[0]

        if len(msg.tags) > 0 and msg.tags[0] == DROP:
            if not msg.tags[0] == DROP:
                return None
    return get_datum(keys, msg.value)


def get_postproc_input(
    keys: list[str], data_path: str, prev_clf_exists=True, prev_model_stale=False
) -> Optional[Datum]:
    thresh_input = get_threshold_input(keys, data_path, prev_model_stale=prev_model_stale)
    _mock_return = return_threshold_clf() if prev_clf_exists else None
    with patch.object(RedisRegistry, "load", Mock(return_value=_mock_return)):
        _out = Threshold().run(keys, thresh_input)
        for msg in _out:
            if POSTPROC_VTX_KEY in msg.tags:
                return get_datum(keys, msg.value)
    return None


def return_mock_lstmae(*_, **__):
    return ArtifactData(
        artifact=LSTMAE(seq_len=12, no_features=2, embedding_dim=4),
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "timestamp": 1653402941,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1645369200000,
            "name": "test::error",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "5",
        },
    )


def return_mock_model_state_dict(seq_len=12, n_features=2) -> ArtifactData:
    return ArtifactData(
        artifact=SparseVanillaAE(seq_len=seq_len, n_features=n_features).state_dict(),
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "timestamp": 1653402941,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1645369200000,
            "name": "test::error",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "5",
        },
    )


def return_stale_model(*_, **__):
    return ArtifactData(
        artifact=SparseVanillaAE(seq_len=12, n_features=2).state_dict(),
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "timestamp": 1653402941,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1656615600000,
            "name": "test::error",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "registry",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "5",
        },
    )


def return_preproc_clf(n_feat=1):
    x = np.random.randn(100, n_feat)
    clf = MinMaxScaler()
    clf.fit(x)
    return ArtifactData(
        artifact=clf,
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1656615600000,
            "name": "test::preproc",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/preproc",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "1",
        },
    )


def return_threshold_clf(n_feat=1):
    x = np.random.randn(100, n_feat)
    clf = StdDevThreshold()
    clf.fit(x)
    return ArtifactData(
        artifact=clf,
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1656615600000,
            "name": "test::thresh",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/thresh",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "1",
        },
    )


def mock_prom_query_metric(*_, **__):
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argorollouts.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
    )


def mock_prom_query_metric2(*_, **__):
    df = pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argorollouts.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
    )
    df.rename(columns={"hash_id": "rollouts_pod_template_hash"}, inplace=True)
    return df


def mock_druid_fetch_data(*_, **__):
    df = pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "druid.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
    )
    return df
