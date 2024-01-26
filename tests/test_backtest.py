import os.path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from numalogic._constants import TESTS_DIR
from numalogic.backtest import PromBacktester
from numalogic.config import NumalogicConf, ModelInfo, TrainerConf, LightningTrainerConf
from numalogic.models.vae import Conv1dVAE

URL = "http://localhost:9090"
CONF = OmegaConf.structured(
    NumalogicConf(
        preprocess=[ModelInfo(name="LogTransformer")],
        model=ModelInfo(name="Conv1dVAE", conf=dict(seq_len=12, n_features=3, latent_dim=1)),
        threshold=ModelInfo(name="RobustMahalanobisThreshold"),
        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(accelerator="cpu", max_epochs=1)),
    )
)


@pytest.fixture
def backtester(tmp_path):
    return PromBacktester(
        url=URL,
        query="namespace_app_rollouts_http_request_error_rate{namespace='sandbox-numalogic-demo'}",
        metrics=[
            "namespace_app_rollouts_cpu_utilization",
            "namespace_app_rollouts_http_request_error_rate",
            "namespace_app_rollouts_memory_utilization",
        ],
        output_dir=tmp_path,
        numalogic_cfg=OmegaConf.to_container(CONF),
    )


@pytest.fixture
def read_data():
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "prom_mv.csv"), index_col="timestamp"
    )


def test_train(backtester, read_data):
    artifacts = backtester.train_models(read_data)
    assert set(artifacts) == {"preproc_clf", "model", "threshold_clf"}
    assert isinstance(artifacts["model"], Conv1dVAE)


def test_scores(backtester, read_data):
    out_df = backtester.generate_scores(read_data)
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.shape[0] == int(backtester.test_ratio * read_data.shape[0])
