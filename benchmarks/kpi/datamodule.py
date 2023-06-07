from typing import Optional
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader

from numalogic.tools.data import TimeseriesDataModule, StreamingDataset


class KPIDataModule(TimeseriesDataModule):
    r"""Data Module to help set up train, test and validation datasets for
    KPI Anomaly detection. This data module splits a single dataset
    into train, validation and test sets using a specified split ratio.

    The dataset can be found in https://github.com/NetManAIOps/KPI-Anomaly-Detection
    Details about the dataset can be found in https://arxiv.org/pdf/2208.03938.pdf

    The dataset is expected to be in the format of:

    |timestamp  | value  | label  | KPI ID |
    |-----------|--------|--------|--------|
    |1476460800| 0.01260 |    0   |da10a69 |

    Args:
    ----
        data_dir: data directory where csv data files are stored
        kpi_idx: index of the KPI to use
        preproc_transforms: list of sklearn compatible preprocessing transformations
        split_ratios: weights of train, validation and test sets respectively
        *args, **kwargs: extra kwargs for TimeseriesDataModule
    """

    def __init__(
        self,
        data_dir: str,
        kpi_idx: int,
        preproc_transforms: Optional[list] = None,
        split_ratios: Sequence[float] = (0.5, 0.2, 0.3),
        *args,
        **kwargs,
    ):
        super().__init__(data=None, *args, **kwargs)

        if len(split_ratios) != 3 or sum(split_ratios) != 1.0:
            raise ValueError("Sum of all the 3 ratios should be 1.0")

        self.split_ratios = split_ratios
        self.data_dir = data_dir
        self.kpi_idx = kpi_idx
        if preproc_transforms:
            if len(preproc_transforms) > 1:
                self.transforms = make_pipeline(preproc_transforms)
            else:
                self.transforms = preproc_transforms[0]
        else:
            self.transforms = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._train_labels = None
        self._val_labels = None
        self._test_labels = None

        self.unique_kpis = None

        self._kpi_df = self.get_kpi_df()

    def _preprocess(self, df: pd.DataFrame) -> npt.NDArray[float]:
        if self.transforms:
            return self.transforms.fit_transform(df[["value"]].to_numpy())
        return df[["value"]].to_numpy()

    def setup(self, stage: str) -> None:
        val_size = np.floor(self.split_ratios[1] * len(self._kpi_df)).astype(int)
        test_size = np.floor(self.split_ratios[2] * len(self._kpi_df)).astype(int)

        if stage == "fit":
            train_df = self._kpi_df[: -(val_size + test_size)]
            val_df = self._kpi_df[val_size:test_size]

            self._train_labels = train_df["label"]
            train_data = self._preprocess(train_df)
            self.train_dataset = StreamingDataset(train_data, self.seq_len)

            self._val_labels = val_df["label"]
            val_data = self._preprocess(val_df)
            self.val_dataset = StreamingDataset(val_data, self.seq_len)

            print(f"Train size: {train_data.shape}\nVal size: {val_data.shape}")

        if stage in ("test", "predict"):
            test_df = self._kpi_df[-test_size:]
            self._test_labels = test_df["label"]
            test_data = self._preprocess(test_df)
            self.test_dataset = StreamingDataset(test_data, self.seq_len)

            print(f"Test size: {test_data.shape}")

    @property
    def val_data(self) -> npt.NDArray[float]:
        return self.val_dataset.data

    @property
    def train_data(self) -> npt.NDArray[float]:
        return self.train_dataset.data

    @property
    def test_data(self) -> npt.NDArray[float]:
        return self.test_dataset.data

    @property
    def val_labels(self) -> npt.NDArray[int]:
        return self._val_labels.to_numpy()

    @property
    def train_labels(self) -> npt.NDArray[int]:
        return self._train_labels.to_numpy()

    @property
    def test_labels(self) -> npt.NDArray[int]:
        return self._test_labels.to_numpy()

    def get_kpi(self, idx: int) -> Optional[str]:
        if self.unique_kpis is not None:
            return self.unique_kpis[idx]
        return None

    def get_kpi_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index(df["timestamp"], inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        self.unique_kpis = df["KPI ID"].unique()
        grouped = df.groupby(["KPI ID", "timestamp"]).sum()
        kpi_id = self.get_kpi(self.kpi_idx)
        print(f"Using KPI ID: {kpi_id}")
        return grouped.loc[kpi_id]

    def val_dataloader(self) -> EVAL_DATALOADERS:
        r"""Creates and returns a DataLoader for the validation
        dataset if validation data is provided.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
