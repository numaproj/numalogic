import datetime
import json
import os
import sys
from unittest import mock
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import pandas as pd
import torch
from mlflow.entities.model_registry import ModelVersion
from numalogic.config import NumalogicConf
from numalogic.models.autoencoder.variants import VanillaAE, LSTMAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.registry import ArtifactData, MLflowRegistry
from omegaconf import OmegaConf
from pynumaflow.function import Datum, Messages
from pynumaflow.function._dtypes import DROP
from sklearn.preprocessing import MinMaxScaler

from anomalydetection._config import PipelineConf, Configs
from anomalydetection._constants import (
    TESTS_DIR,
    POSTPROC_VTX_KEY,
    TESTS_RESOURCES,
    CONFIG_DIR,
)
from anomalydetection.factory import HandlerFactory


sys.modules["anomalydetection.mlflow"] = MagicMock()
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")


def mockenv(**envvars):
    return mock.patch.dict(os.environ, envvars, clear=True)


def get_datum(data: str or bytes) -> Datum:
    if type(data) is not bytes:
        data = json.dumps(data).encode("utf-8")

    return Datum(
        key="random_key",
        value=data,
        event_time=datetime.datetime.now(),
        watermark=datetime.datetime.now(),
    )


def get_stream_data(data_path: str):
    with open(data_path) as fp:
        data = json.load(fp)
    return data


def get_prepoc_input(data_path: str) -> Messages:
    out = Messages()
    data = get_stream_data(data_path)
    for obj in data:
        handler_ = HandlerFactory.get_handler("window")
        _out = handler_("", get_datum(obj))
        if _out.items()[0].key != DROP:
            out.append(_out.items()[0])
    return out


def get_inference_input(data_path: str, prev_clf_exists=True) -> Messages:
    out = Messages()
    preproc_input = get_prepoc_input(data_path)
    _mock_return = return_preproc_clf() if prev_clf_exists else None
    with patch.object(MLflowRegistry, "load", Mock(return_value=_mock_return)):
        for msg in preproc_input.items():
            _in = get_datum(msg.value)
            handler_ = HandlerFactory.get_handler("preprocess")
            _out = handler_("", _in)
            if _out.items()[0].key != DROP:
                out.append(_out.items()[0])
    return out


def get_threshold_input(data_path: str, prev_clf_exists=True, prev_model_stale=False) -> Messages:
    out = Messages()
    inference_input = get_inference_input(data_path)
    if prev_model_stale:
        _mock_return = return_stale_model()
    elif prev_clf_exists:
        _mock_return = return_mock_lstmae()
    else:
        _mock_return = None
    with patch.object(MLflowRegistry, "load", Mock(return_value=_mock_return)):
        for msg in inference_input.items():
            _in = get_datum(msg.value)
            handler_ = HandlerFactory.get_handler("inference")
            _out = handler_(None, _in)
            if _out.items()[0].key != DROP:
                out.append(_out.items()[0])
    return out


def get_postproc_input(data_path: str, prev_clf_exists=True, prev_model_stale=False) -> Messages:
    out = Messages()
    thresh_input = get_threshold_input(data_path, prev_model_stale=prev_model_stale)
    _mock_return = return_threshold_clf() if prev_clf_exists else None
    with patch.object(MLflowRegistry, "load", Mock(return_value=_mock_return)):
        for msg in thresh_input.items():
            _in = get_datum(msg.value)
            handler_ = HandlerFactory.get_handler("threshold")
            _out = handler_(None, _in)
            for _msg in _out.items():
                if _msg.key == bytes(POSTPROC_VTX_KEY, "utf-8"):
                    out.append(_msg)
    return out


def return_mock_vanilla(*_, **__):
    return {
        "primary_artifact": VanillaAE(2),
        "metadata": torch.load(os.path.join(MODEL_DIR, "model_cpu.pth")),
        "model_properties": ModelVersion(
            creation_timestamp=1656615600000,
            current_stage="Production",
            description="",
            last_updated_timestamp=datetime.datetime.now().timestamp() * 1000,
            name="sandbox_numalogic_demo:metric_1",
            run_id="6f1e582fb6194bbdaa4141feb2ce8e27",
            run_link="",
            source="mlflow-artifacts:/0/6f1e582fb6194bbdaa4141feb2ce8e27/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="125",
        ),
    }


def return_mock_lstmae(*_, **__):
    return ArtifactData(
        artifact=LSTMAE(seq_len=2, no_features=1, embedding_dim=4),
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
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
        artifact=VanillaAE(seq_len=2),
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1656615600000,
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


def mock_argocd_query_metric(*_, **__):
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argocd.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
    )


def mock_rollout_query_metric(*_, **__):
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argorollouts.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
    )


def mock_rollout_query_metric2(*_, **__):
    df = pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argorollouts.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
    )
    df.rename(columns={"hash_id": "rollouts_pod_template_hash"}, inplace=True)
    return df


def mock_configs():
    schema: Configs = OmegaConf.structured(Configs)

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "config.yaml"))
    user_configs = OmegaConf.merge(schema, conf).configs

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "default-config.yaml"))
    default_configs = OmegaConf.merge(schema, conf).configs

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "numalogic_config.yaml"))
    schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
    default_numalogic = OmegaConf.merge(schema, conf)

    conf = OmegaConf.load(os.path.join(CONFIG_DIR, "default-configs", "pipeline_config.yaml"))
    schema: PipelineConf = OmegaConf.structured(PipelineConf)
    pipeline_config = OmegaConf.merge(schema, conf)

    return user_configs, default_configs, default_numalogic, pipeline_config
