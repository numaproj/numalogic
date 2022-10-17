import logging
from typing import Sequence, Callable, Optional, Union, BinaryIO

import numpy as np
import pandas as pd
from ml_steps.pl_factory import ModelPlFactory
from numpy.typing import NDArray
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from torch import nn

LOGGER = logging.getLogger(__name__)


class SimpleMLPipeline:
    """
    Inference Pipeline.
    """

    def __init__(
        self,
        metric: str,
        preprocess_steps: Sequence[TransformerMixin] = None,
        postprocess_funcs: Sequence[Callable] = None,
        model_plname="ae",
        **model_pl_kw
    ):
        self.metric = metric

        if model_pl_kw:
            self.model_ppl = ModelPlFactory.get_pl_obj(model_plname, **model_pl_kw)
        else:
            self.model_ppl = None

        self.preprocess_pipeline = make_pipeline(*preprocess_steps) if preprocess_steps else None
        self.postprocess_funcs = postprocess_funcs or []

    @property
    def model(self) -> Optional[nn.Module]:
        return self.model_ppl.model

    @staticmethod
    def clean_data(df: pd.DataFrame, limit=12):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(method="ffill", limit=limit)
        df = df.fillna(method="bfill", limit=limit)
        if df.columns[df.isna().any()].tolist():
            df.dropna(inplace=True)
        return df

    def preprocess(self, X: NDArray, train=True) -> NDArray[float]:
        if not self.preprocess_pipeline:
            LOGGER.warning("No preprocess steps provided.")
            return X
        if train:
            return self.preprocess_pipeline.fit_transform(X)
        return self.preprocess_pipeline.transform(X)

    def train(self, X: NDArray) -> None:
        """
        Infer/predict on the given data.
        Note: this assumes that X is already preprocessed.
        :param X: Numpy Array
        """
        if not self.model_ppl:
            raise ValueError("Model pipeline is not initialized.")
        self.model_ppl.fit(X)

    def infer(self, X: NDArray) -> NDArray[float]:
        """
        Infer/predict on the given data.
        Note: this assumes that X is already preprocessed.
        :param X: Numpy Array
        :return: Anomaly scores
        """
        if not self.model_ppl:
            raise ValueError("Model pipeline is not initialized.")
        return self.model_ppl.score(X)

    def postprocess(self, y: NDArray) -> NDArray[float]:
        for func in self.postprocess_funcs:
            y = func(np.copy(y))
        return y

    def load_model(
        self, path_or_buf: Union[str, BinaryIO] = None, model: nn.Module = None, **metadata
    ) -> None:
        if not self.model_ppl:
            raise ValueError(
                "An initialized model pipeline object is required for loading a saved model."
            )
        self.model_ppl.load(path=path_or_buf, model=model, **metadata)

    def save_model(self, path: Union[str, None] = None) -> Optional[BinaryIO]:
        if not self.model_ppl:
            raise ValueError("Model pipeline is not initialized.")
        return self.model_ppl.save(path)
