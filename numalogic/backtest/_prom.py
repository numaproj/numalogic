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
from numpy.typing import NDArray
from omegaconf import OmegaConf

from numalogic._constants import BASE_DIR
from numalogic.backtest._constants import DEFAULT_SEQUENCE_LEN
from numalogic.config import (
    NumalogicConf,
    ModelFactory,
    PreprocessFactory,
    PostprocessFactory,
    ThresholdFactory,
)
from numalogic.connectors import ConnectorType
from numalogic.connectors.prometheus import PrometheusFetcher
from numalogic.tools.data import StreamingDataset, inverse_window
from numalogic.tools.types import artifact_t
from numalogic.udfs import UDFFactory, StreamConf, MLPipelineConf

DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, ".btoutput")
LOGGER = logging.getLogger(__name__)


class PromBacktester:
    def __init__(
        self,
        url: str,
        query: str,
        return_labels: Optional[list[str]] = None,
        metrics: Optional[list[str]] = None,
        lookback_days: int = 8,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        test_ratio: float = 0.25,
        numalogic_cfg: Optional[dict] = None,
        load_saved_conf: bool = False,
        experiment_name: str = "exp",
    ):
        self._url = url
        self.test_ratio = test_ratio
        self.lookback_days = lookback_days
        self.return_labels = return_labels

        self.out_dir = self.get_outdir(experiment_name, outdir=output_dir)
        self._datapath = os.path.join(self.out_dir, "data.csv")
        self._modelpath = os.path.join(self.out_dir, "models.pt")
        self._outpath = os.path.join(self.out_dir, "output.csv")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.query = query
        self.metrics = metrics or []
        self.conf: StreamConf = self._init_conf(metrics, numalogic_cfg, load_saved_conf)
        self.nlconf: NumalogicConf = self.conf.get_numalogic_conf()

    def _init_conf(self, metrics: list[str], nl_conf: dict, load_saved_conf: bool) -> StreamConf:
        if load_saved_conf:
            try:
                nl_conf = OmegaConf.load(os.path.join(self.out_dir, "config.yaml"))
            except FileNotFoundError:
                LOGGER.warning("No saved config found in %s", self.out_dir)
            else:
                LOGGER.info("Loaded saved config from %s", self.out_dir)

        if nl_conf:
            LOGGER.info("Using provided config!")
            return StreamConf(
                source=ConnectorType.prometheus,
                window_size=DEFAULT_SEQUENCE_LEN,
                ml_pipelines={
                    "default": MLPipelineConf(
                        pipeline_id="default",
                        metrics=metrics,
                        numalogic_conf=OmegaConf.to_object(
                            OmegaConf.merge(
                                OmegaConf.structured(NumalogicConf), OmegaConf.create(nl_conf)
                            ),
                        ),
                    )
                },
            )

        raise ValueError("Provide one of numalogic_conf or load_saved_conf")

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

        if self.metrics:
            df = df[self.metrics]

        df_train, _ = self._split_data(df)

        x_train = df_train.to_numpy(dtype=np.float32)
        LOGGER.info("Training data shape: %s", x_train.shape)

        artifacts = UDFFactory.get_udf_cls("promtrainer").compute(
            model=ModelFactory().get_instance(self.nlconf.model),
            input_=x_train,
            preproc_clf=PreprocessFactory().get_pipeline_instance(self.nlconf.preprocess),
            threshold_clf=ThresholdFactory().get_instance(self.nlconf.threshold),
            numalogic_cfg=self.nlconf,
        )
        artifacts_dict = {
            "model": artifacts["inference"].artifact,
            "preproc_clf": artifacts["preproc_clf"].artifact,
            "threshold_clf": artifacts["threshold_clf"].artifact,
        }
        with open(self._modelpath, "wb") as f:
            torch.save(artifacts_dict, f)

        with open(os.path.join(self.out_dir, "config.yaml"), "w") as f:
            OmegaConf.save(self.nlconf, f)

        LOGGER.info("Models saved in %s", self._modelpath)
        return artifacts_dict

    def generate_scores(
        self,
        df: pd.DataFrame,
        model_path: Optional[str] = None,
        use_full_data: bool = False,
    ) -> pd.DataFrame:
        """
        Generate scores for the given data.

        Args:
        -------
            df: Dataframe with timestamp and metric values
            model_path: Path to the saved models
            use_full_data: If True, use the full data for generating scores else use only the test

        Returns
        -------
            Dataframe with timestamp and metric values

        Raises
        ------
            RuntimeError: If valid model is not provided when use_full_data is True
        """
        try:
            artifacts = self._load_or_train_model(df, model_path, avoid_training=use_full_data)
        except FileNotFoundError as err:
            raise RuntimeError(
                "Valid model needs to be provided if use_full_data is True!"
            ) from err

        if use_full_data:
            df_test = df[self.metrics]
        else:
            _, df_test = self._split_data(df[self.metrics])
        x_test = df_test.to_numpy(dtype=np.float32)
        LOGGER.info("Test data shape: %s", df_test.shape)

        preproc_udf = UDFFactory.get_udf_cls("preprocess")
        nn_udf = UDFFactory.get_udf_cls("inference")
        postproc_udf = UDFFactory.get_udf_cls("postprocess")

        # Preprocess
        x_scaled = preproc_udf.compute(model=artifacts["preproc_clf"], input_=x_test)

        ds = StreamingDataset(x_scaled, seq_len=self.conf.window_size)
        raw_scores = np.zeros((len(ds), self.conf.window_size), dtype=np.float32)
        final_scores = np.zeros_like(raw_scores, dtype=np.float32)
        postproc_func = PostprocessFactory().get_instance(self.nlconf.postprocess)

        x_recon = np.zeros((len(ds), self.conf.window_size, len(self.metrics)), dtype=np.float32)

        # Model Inference
        for idx, arr in enumerate(ds):
            x_recon[idx] = nn_udf.compute(model=artifacts["model"], input_=arr)
            raw_scores[idx], final_scores[idx] = postproc_udf.compute(
                model=artifacts["threshold_clf"],
                input_=x_recon[idx],
                postproc_clf=postproc_func,
            )

        x_recon = inverse_window(torch.from_numpy(x_recon), method="keep_first").numpy()
        raw_scores = inverse_window(
            torch.unsqueeze(torch.from_numpy(raw_scores), dim=2), method="keep_first"
        ).numpy()
        final_scores = inverse_window(
            torch.unsqueeze(torch.from_numpy(final_scores), dim=2), method="keep_first"
        ).numpy()

        return self._construct_output(
            df_test,
            preproc_out=x_scaled,
            nn_out=x_recon,
            thresh_out=raw_scores,
            postproc_out=final_scores,
        )

    @classmethod
    def get_outdir(cls, expname: str, outdir=DEFAULT_OUTPUT_DIR) -> str:
        """Get the output directory for the given metric."""
        return os.path.join(outdir, expname)

    def read_data(self, fill_na_value: float = 0.0, save=True) -> pd.DataFrame:
        datafetcher = PrometheusFetcher(self._url)
        df = datafetcher.raw_fetch(
            query=self.query,
            start=(datetime.now() - timedelta(days=self.lookback_days)),
            end=datetime.now(),
            return_labels=self.return_labels,
        )
        LOGGER.info(
            "Fetched dataframe with lookback days: %s with shape: %s", self.lookback_days, df.shape
        )
        if df.empty:
            return df
        if self.metrics:
            df = df[self.metrics]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(fill_na_value)
        if save:
            df.to_csv(self._datapath, index=True)
        return df

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

    def _load_or_train_model(
        self, df: pd.DataFrame, model_path: str, avoid_training: bool = False
    ) -> dict[str, artifact_t]:
        _modelpath = model_path or self._modelpath
        try:
            with open(_modelpath, "rb") as f:
                artifacts = torch.load(f)
        except FileNotFoundError:
            if avoid_training:
                raise
            LOGGER.info("No saved models found! Training models...")
            artifacts = self.train_models(df)
        else:
            LOGGER.info("Loaded models from %s", _modelpath)
        return artifacts

    def _construct_output(
        self,
        input_df: pd.DataFrame,
        preproc_out: NDArray[float],
        nn_out: NDArray[float],
        thresh_out: NDArray[float],
        postproc_out: NDArray[float],
    ) -> pd.DataFrame:
        ts_idx = input_df.index
        dfs = {
            "input": input_df,
            "preproc_out": pd.DataFrame(
                preproc_out,
                columns=self.metrics,
                index=ts_idx,
            ),
            "model_out": pd.DataFrame(
                nn_out,
                columns=self.metrics,
                index=ts_idx,
            ),
            "thresh_out": pd.DataFrame(
                thresh_out,
                columns=["unified_score"],
                index=ts_idx,
            ),
            "postproc_out": pd.DataFrame(
                postproc_out,
                columns=["unified_score"],
                index=ts_idx,
            ),
        }
        return pd.concat(dfs, axis=1)