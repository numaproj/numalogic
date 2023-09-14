# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os.path
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from numalogic._constants import BASE_DIR
from numalogic.backtest._constants import DEFAULT_SEQUENCE_LEN
from numalogic.config import (
    TrainerConf,
    LightningTrainerConf,
    NumalogicConf,
    ModelInfo,
    ModelFactory,
    PreprocessFactory,
    PostprocessFactory,
    ThresholdFactory,
)
from numalogic.connectors import ConnectorType
from numalogic.connectors.prometheus import PrometheusFetcher
from numalogic.tools.data import StreamingDataset, inverse_window
from numalogic.tools.types import artifact_t
from numalogic.udfs import UDFFactory, StreamConf

DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, ".btoutput")
LOGGER = logging.getLogger(__name__)


def _init_default_streamconf(metrics: list[str]) -> StreamConf:
    numalogic_cfg = NumalogicConf(
        model=ModelInfo(
            "VanillaAE", conf={"seq_len": DEFAULT_SEQUENCE_LEN, "n_features": len(metrics)}
        ),
        preprocess=[ModelInfo("StandardScaler")],
        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(accelerator="cpu")),
    )
    return StreamConf(
        source=ConnectorType.prometheus,
        window_size=DEFAULT_SEQUENCE_LEN,
        metrics=metrics,
        numalogic_conf=numalogic_cfg,
    )


class PromUnivarBacktester:
    """
    Class for running backtest for a single metric on data from Prometheus or Thanos.

    Args:
        url: Prometheus/Thanos URL
        namespace: Namespace of the metric
        appname: Application name
        metric: Metric name
        return_labels: Prometheus label names as columns to return
        lookback_days: Number of days of data to fetch
        output_dir: Output directory
        test_ratio: Ratio of test data to total data
        stream_conf: Stream configuration
    """

    def __init__(
        self,
        url: str,
        namespace: str,
        appname: str,
        metric: str,
        return_labels: Optional[list[str]] = None,
        lookback_days: int = 8,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        test_ratio: float = 0.25,
        stream_conf: Optional[StreamConf] = None,
    ):
        self._url = url
        self.namespace = namespace
        self.appname = appname
        self.metric = metric
        self.conf = stream_conf or _init_default_streamconf([metric])
        self.test_ratio = test_ratio
        self.lookback_days = lookback_days
        self.return_labels = return_labels

        self._seq_len = self.conf.window_size
        self._n_features = len(self.conf.metrics)

        self.out_dir = self.get_outdir(appname, metric, outdir=output_dir)
        self._datapath = os.path.join(self.out_dir, "data.csv")
        self._modelpath = os.path.join(self.out_dir, "models.pt")
        self._outpath = os.path.join(self.out_dir, "output.csv")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    @classmethod
    def get_outdir(cls, appname: str, metric: str, outdir=DEFAULT_OUTPUT_DIR) -> str:
        """Get the output directory for the given metric."""
        if not appname:
            return os.path.join(outdir, metric)
        _key = ":".join([appname, metric])
        return os.path.join(outdir, _key)

    def read_data(self, fill_na_value: float = 0.0) -> pd.DataFrame:
        """
        Reads data from Prometheus/Thanos and returns a dataframe.

        Args:
            fill_na_value: Value to fill NaNs with

        Returns
        -------
            Dataframe with timestamp and metric values
        """
        datafetcher = PrometheusFetcher(self._url)
        df = datafetcher.fetch(
            metric_name=self.metric,
            start=(datetime.now() - timedelta(days=self.lookback_days)),
            end=datetime.now(),
            filters={"namespace": self.namespace, "app": self.appname},
            return_labels=self.return_labels,
            aggregate=False,
        )
        LOGGER.info(
            "Fetched dataframe with lookback days: %s with shape: %s", self.lookback_days, df.shape
        )

        df.set_index(["timestamp"], inplace=True)
        df.index = pd.to_datetime(df.index)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(fill_na_value)

        df.to_csv(self._datapath, index=True)
        return df

    def train_models(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> dict[str, artifact_t]:
        """
        Train models for the given data.

        Args:
            df: Dataframe with timestamp and metric values

        Returns
        -------
            Dictionary of trained models
        """
        if df is None:
            df = self._read_or_fetch_data()

        df_train, _ = self._split_data(df[[self.metric]])
        x_train = df_train.to_numpy(dtype=np.float32)
        LOGGER.info("Training data shape: %s", x_train.shape)

        artifacts = UDFFactory.get_udf_cls("trainer").compute(
            model=ModelFactory().get_instance(self.conf.numalogic_conf.model),
            input_=x_train,
            preproc_clf=PreprocessFactory().get_pipeline_instance(
                self.conf.numalogic_conf.preprocess
            ),
            threshold_clf=ThresholdFactory().get_instance(self.conf.numalogic_conf.threshold),
            trainer_cfg=self.conf.numalogic_conf.trainer,
        )
        with open(self._modelpath, "wb") as f:
            torch.save(artifacts, f)
        LOGGER.info("Models saved in %s", self._modelpath)
        return artifacts

    def generate_scores(
        self, df: Optional[pd.DataFrame] = None, model_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate scores for the given data.

        Args:
            df: Dataframe with timestamp and metric values
            model_path: Path to the saved models

        Returns
        -------
            Dataframe with timestamp and metric values
        """
        if df is None:
            df = self._read_or_fetch_data()

        artifacts = self._load_or_train_model(df, model_path)

        _, df_test = self._split_data(df[[self.metric]])
        x_test = df_test.to_numpy(dtype=np.float32)
        LOGGER.info("Test data shape: %s", df_test.shape)

        preproc_udf = UDFFactory.get_udf_cls("preprocess")
        nn_udf = UDFFactory.get_udf_cls("inference")
        postproc_udf = UDFFactory.get_udf_cls("postprocess")

        x_scaled = preproc_udf.compute(model=artifacts["preproc_clf"], input_=x_test)

        ds = StreamingDataset(x_scaled, seq_len=self.conf.window_size)
        anomaly_scores = np.zeros(
            (len(ds), self.conf.window_size, len(self.conf.metrics)), dtype=np.float32
        )
        x_recon = np.zeros_like(anomaly_scores, dtype=np.float32)
        postproc_func = PostprocessFactory().get_instance(self.conf.numalogic_conf.postprocess)

        for idx, arr in enumerate(ds):
            x_recon[idx] = nn_udf.compute(model=artifacts["model"], input_=arr)
            anomaly_scores[idx] = postproc_udf.compute(
                model=artifacts["threshold_clf"],
                input_=x_recon[idx],
                postproc_clf=postproc_func,
            )
        x_recon = inverse_window(torch.from_numpy(x_recon)).numpy()
        final_scores = np.mean(anomaly_scores, axis=1)
        output_df = self._construct_output_df(
            timestamps=df_test.index,
            test_data=x_test,
            preproc_out=x_scaled,
            nn_out=x_recon,
            postproc_out=final_scores,
        )
        output_df.to_csv(self._outpath, index=True, index_label="timestamp")
        LOGGER.info("Results saved in: %s", self._outpath)
        return output_df

    def save_plots(
        self, output_df: Optional[pd.DataFrame] = None, plotname: str = "plot.png"
    ) -> None:
        """
        Save plots for the given data.

        Args:
            output_df: Dataframe with timestamp, and anomaly scores
            plotname: Name of the plot file
        """
        if output_df is None:
            output_df = pd.read_csv(self._outpath, index_col="timestamp", parse_dates=True)

        fig, axs = plt.subplots(4, 1, sharex="col", figsize=(15, 8))

        axs[0].plot(output_df["metric"], color="b")
        axs[0].set_ylabel("Original metric")
        axs[0].grid(True)
        axs[0].set_title(
            f"TEST SET RESULTS\nMetric: {self.metric}\n"
            f"namespace: {self.namespace}\napp: {self.appname}"
        )

        axs[1].plot(output_df["preprocessed"], color="g")
        axs[1].grid(True)
        axs[1].set_ylabel("Preprocessed metric")

        axs[2].plot(output_df["model_out"], color="black")
        axs[2].grid(True)
        axs[2].set_ylabel("NN model output")

        axs[3].plot(output_df["scores"], color="r")
        axs[3].grid(True)
        axs[3].set_ylabel("Anomaly Score")
        axs[3].set_xlabel("Time")
        axs[3].set_ylim(0, 10)

        fig.tight_layout()
        _fname = os.path.join(self.out_dir, plotname)
        fig.savefig(_fname)
        LOGGER.info("Plot file: %s saved in %s", plotname, self.out_dir)

    def _split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_size = int(df.shape[0] * self.test_ratio)
        return df.iloc[:-test_size], df.iloc[-test_size:]

    def _read_or_fetch_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self._datapath, dtype=np.float32)
        except FileNotFoundError:
            LOGGER.info("No saved data found! Fetching data...")
            df = self.read_data()
        else:
            LOGGER.info("Saved data found! Reading from %s", self._datapath)
        df.set_index(["timestamp"], inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def _load_or_train_model(self, df: pd.DataFrame, model_path: str) -> dict[str, artifact_t]:
        _modelpath = model_path or self._modelpath
        try:
            with open(_modelpath, "rb") as f:
                artifacts = torch.load(f)
        except FileNotFoundError:
            LOGGER.info("No saved models found! Training models...")
            artifacts = self.train_models(df)
        else:
            LOGGER.info("Loaded models from %s", _modelpath)
        return artifacts

    def _construct_output_df(
        self,
        timestamps: pd.Index,
        test_data: NDArray[float],
        preproc_out: NDArray[float],
        nn_out: NDArray[float],
        postproc_out: NDArray[float],
    ) -> pd.DataFrame:
        scores = np.vstack(
            [
                np.full((self._seq_len - 1, self._n_features), fill_value=np.nan),
                postproc_out,
            ]
        )

        return pd.DataFrame(
            {
                "metric": test_data.squeeze(),
                "preprocessed": preproc_out.squeeze(),
                "model_out": nn_out.squeeze(),
                "scores": scores.squeeze(),
            },
            index=timestamps,
        )
