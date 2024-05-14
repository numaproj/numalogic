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
from dataclasses import dataclass
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


@dataclass
class OutDataFrames:
    input: pd.DataFrame
    preproc_out: pd.DataFrame
    model_out: pd.DataFrame
    thresh_out: pd.DataFrame
    postproc_out: pd.DataFrame
    unified_out: pd.DataFrame
    static_out: Optional[pd.DataFrame] = None
    static_features: Optional[pd.DataFrame] = None
    adjusted_unified: Optional[pd.DataFrame] = None


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
        self.seq_len = self.conf.window_size

    def _init_conf(self, metrics: list[str], nl_conf: dict, load_saved_conf: bool) -> StreamConf:
        if load_saved_conf:
            try:
                nl_conf = OmegaConf.load(os.path.join(self.out_dir, "config.yaml"))
            except FileNotFoundError:
                LOGGER.warning("No saved config found in %s", self.out_dir)
            else:
                LOGGER.info("Loaded saved config from %s", self.out_dir)

        if not nl_conf:
            raise ValueError("Provide one of numalogic_conf or load_saved_conf")

        nl_conf: NumalogicConf = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(NumalogicConf), OmegaConf.create(nl_conf)),
        )

        return StreamConf(
            source=ConnectorType.prometheus,
            window_size=nl_conf.model.conf.get("seq_len") or DEFAULT_SEQUENCE_LEN,
            ml_pipelines={
                "default": MLPipelineConf(
                    pipeline_id="default", metrics=metrics, numalogic_conf=nl_conf
                )
            },
        )

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

        if self.nlconf.trainer.transforms:
            train_txs = PreprocessFactory().get_pipeline_instance(self.nlconf.trainer.transforms)
        else:
            train_txs = None
        artifacts = UDFFactory.get_udf_cls("promtrainer").compute(
            model=ModelFactory().get_instance(self.nlconf.model),
            input_=x_train,
            trainer_transform=train_txs,
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

    def window_inverse(self, x: NDArray[float]) -> NDArray[float]:
        """
        Perform inverse windowing on the given data.

        If stride is 1, return the data as is.
        If stride is 2, return the data by stacking the two windows.
        Stride > 2 is not supported yet.

        Args:
        -------
            x: Input data

        Returns
        -------
            Inverse windowed data with feature recovery if stride is 2
        """
        x = torch.from_numpy(x)
        stride = self.nlconf.trainer.ds_stride

        if stride == 1:
            return inverse_window(x).numpy()

        # TODO support for stride > 2
        if stride > 2:
            raise NotImplementedError("Stride > 2 not supported!")

        # Recover the features
        x1, x2 = x[:, ::stride], x[:, 1::stride]
        x1 = inverse_window(x1).numpy()
        x2 = inverse_window(x2).numpy()
        return np.hstack([x1, x2])

    def generate_scores(
        self,
        df: pd.DataFrame,
        model_path: Optional[str] = None,
        use_full_data: bool = False,
    ) -> OutDataFrames:
        """
        Generate scores for the given data.

        Args:
        -------
            df: Dataframe with timestamp and metric values
            model_path: Path to the saved models
            use_full_data: If True, use the full data for generating scores else use only the test

        Returns
        -------
            Dict of dataframes with timestamp and metric values for each step

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

        n_feat = x_scaled.shape[1]

        ds = StreamingDataset(x_scaled, seq_len=self.seq_len, stride=self.nlconf.trainer.ds_stride)

        x_recon = np.zeros((len(ds), self.seq_len, n_feat), dtype=np.float32)
        raw_scores = np.zeros((len(ds), self.seq_len, n_feat), dtype=np.float32)
        unified_raw_scores = np.zeros((len(ds), 1), dtype=np.float32)

        feature_scores = np.zeros((len(ds), n_feat), dtype=np.float32)
        unified_scores = np.zeros((len(ds), 1), dtype=np.float32)

        postproc_func = PostprocessFactory().get_instance(self.nlconf.postprocess)

        # Model Inference
        for idx, arr in enumerate(ds):
            x_recon[idx] = nn_udf.compute(model=artifacts["model"], input_=arr)

            # TODO support for multivariate thresholding functions
            thresh_out = postproc_udf.compute_threshold(artifacts["threshold_clf"], x_recon[idx])
            raw_scores[idx] = thresh_out

            winscores = postproc_udf.compute_feature_scores(
                raw_scores[idx], self.nlconf.score.window_agg
            )  # (nfeat,)

            unified_raw_scores[idx] = postproc_udf.compute_unified_score(
                winscores,
                feat_agg_conf=self.nlconf.score.feature_agg,
            )

            feature_scores[idx] = postproc_udf.compute_postprocess(postproc_func, winscores)

            unified_scores[idx] = postproc_udf.compute_postprocess(
                postproc_func, unified_raw_scores[idx]
            )

        x_recon = self.window_inverse(x_recon)
        raw_scores = self.window_inverse(raw_scores)

        feature_scores = np.vstack(
            [
                np.full((len(x_test) - len(ds), n_feat), fill_value=np.nan),
                feature_scores,
            ]
        )
        unified_scores = np.vstack(
            [np.full((len(x_test) - len(ds), 1), fill_value=np.nan), unified_scores]
        )

        out_dfs = self._construct_output(
            df_test,
            preproc_out=x_scaled,
            nn_out=x_recon,
            thresh_out=raw_scores,
            postproc_out=feature_scores,
            unified_out=unified_scores,
        )
        if self.nlconf.score.adjust:
            static_scores = self.generate_static_scores(df_test)
            out_dfs.static_out = static_scores["static_unified"]
            out_dfs.static_features = static_scores["static_features"]

            out_dfs.adjusted_unified = pd.concat(
                [out_dfs.unified_out, out_dfs.static_out], axis=1
            ).max(axis=1)

        return out_dfs

    def generate_static_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.nlconf.score.adjust:
            raise ValueError("No adjust params provided in the config!")

        metrics = list(self.nlconf.score.adjust.upper_limits)
        x_test = df[metrics].to_numpy(dtype=np.float32)

        postproc_udf = UDFFactory.get_udf_cls("postprocess")
        ds = StreamingDataset(x_test, seq_len=self.seq_len)

        feature_scores = np.zeros((len(ds), len(metrics)), dtype=np.float32)
        unified_scores = np.zeros((len(ds), 1), dtype=np.float32)

        for idx, arr in enumerate(ds):
            feature_scores[idx] = postproc_udf.compute_static_threshold(
                arr, score_conf=self.nlconf.score
            )
            unified_scores[idx] = postproc_udf.compute_unified_score(
                feature_scores[idx], feat_agg_conf=self.nlconf.score.adjust.feature_agg
            )
        feature_scores = np.vstack(
            [
                np.full((len(x_test) - len(ds), len(metrics)), fill_value=np.nan),
                feature_scores,
            ]
        )
        unified_scores = np.vstack(
            [np.full((len(x_test) - len(ds), 1), fill_value=np.nan), unified_scores]
        )
        dfs = {
            "input": df,
            "static_features": pd.DataFrame(
                feature_scores,
                columns=metrics,
                index=df.index,
            ),
            "static_unified": pd.DataFrame(
                unified_scores,
                columns=["unified"],
                index=df.index,
            ),
        }
        return pd.concat(dfs, axis=1)

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
        unified_out: NDArray[float],
    ) -> OutDataFrames:
        ts_idx = input_df.index

        if thresh_out.shape[1] > 1:
            thresh_df = pd.DataFrame(
                thresh_out,
                columns=self.metrics,
                index=ts_idx,
            )
        else:
            thresh_df = pd.DataFrame(
                thresh_out,
                columns=["unified"],
                index=ts_idx,
            )

        if postproc_out.shape[1] > 1:
            postproc_df = pd.DataFrame(
                postproc_out,
                columns=self.metrics,
                index=ts_idx,
            )
        else:
            postproc_df = pd.DataFrame(
                postproc_out,
                columns=["unified"],
                index=ts_idx,
            )

        if len(preproc_out) == len(ts_idx) and preproc_out.shape[1] == len(self.metrics):
            preproc_df = pd.DataFrame(
                preproc_out,
                columns=self.metrics,
                index=ts_idx,
            )
        else:
            preproc_df = pd.DataFrame(
                preproc_out,
            )

        return OutDataFrames(
            input=input_df,
            preproc_out=preproc_df,
            model_out=pd.DataFrame(
                nn_out,
                columns=self.metrics,
                index=ts_idx,
            ),
            thresh_out=thresh_df,
            postproc_out=postproc_df,
            unified_out=pd.DataFrame(
                unified_out,
                columns=["unified"],
                index=ts_idx,
            ),
        )
