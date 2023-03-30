import os
import unittest

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.models.autoencoder.variants import (
    Conv1dAE,
    LSTMAE,
    SparseVanillaAE,
    TransformerAE,
    SparseConv1dAE,
)
from numalogic.tools.data import TimeseriesDataModule, StreamingDataset

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
EPOCHS = 5
BATCH_SIZE = 256
SEQ_LEN = 12
LR = 0.001
torch.manual_seed(42)


class TestAutoencoderTrainer(unittest.TestCase):
    x_train = None
    x_val = None
    x_test = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]

        scaler = StandardScaler()
        cls.x_train = scaler.fit_transform(df[:-480])
        cls.x_val = scaler.transform(df[-480:-240])
        cls.x_test = scaler.transform(df[-240:])

        print(cls.x_train.shape, cls.x_val.shape, cls.x_test.shape)

    def test_trainer_01(self):
        model = Conv1dAE(seq_len=SEQ_LEN, in_channels=self.x_train.shape[1], enc_channels=(4, 8))
        datamodule = TimeseriesDataModule(
            SEQ_LEN, self.x_train, val_data=self.x_val, batch_size=BATCH_SIZE
        )
        trainer = AutoencoderTrainer(max_epochs=5, enable_progress_bar=True, limit_val_batches=1)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.x_test, SEQ_LEN), batch_size=1)
        y_test = trainer.predict(model, dataloaders=streamloader, unbatch=True)
        self.assertTupleEqual(self.x_test.shape, y_test.size())

    def test_trainer_02(self):
        model = Conv1dAE(seq_len=SEQ_LEN, in_channels=self.x_train.shape[1], enc_channels=(4,))
        datamodule = TimeseriesDataModule(SEQ_LEN, self.x_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(max_epochs=5, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        y_train = trainer.predict(model, dataloaders=datamodule.train_dataloader())
        self.assertTupleEqual(self.x_train.shape, y_train.size())

        streamloader = DataLoader(StreamingDataset(self.x_test, SEQ_LEN), batch_size=BATCH_SIZE)
        y_test_batched = trainer.predict(model, dataloaders=streamloader, unbatch=False)
        self.assertTupleEqual((229, SEQ_LEN, self.x_test.shape[1]), y_test_batched.size())

    def test_trainer_03(self):
        model = LSTMAE(seq_len=SEQ_LEN, no_features=self.x_train.shape[1], embedding_dim=4)
        datamodule = TimeseriesDataModule(
            SEQ_LEN, self.x_train, val_data=self.x_val, batch_size=BATCH_SIZE
        )
        trainer = AutoencoderTrainer(max_epochs=20, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.x_test, SEQ_LEN), batch_size=16)
        y_test = trainer.predict(model, dataloaders=streamloader, unbatch=True)
        self.assertTupleEqual(self.x_test.shape, y_test.size())

    def test_trainer_04(self):
        model = SparseVanillaAE(seq_len=SEQ_LEN, n_features=self.x_train.shape[1])
        datamodule = TimeseriesDataModule(
            SEQ_LEN, self.x_train, val_data=self.x_val, batch_size=BATCH_SIZE
        )
        trainer = AutoencoderTrainer(max_epochs=15, enable_progress_bar=True, limit_val_batches=1)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.x_test, SEQ_LEN))
        y_test = trainer.predict(model, dataloaders=streamloader, unbatch=True)
        self.assertTupleEqual(self.x_test.shape, y_test.size())

    def test_trainer_05(self):
        model = TransformerAE(seq_len=SEQ_LEN, n_features=self.x_train.shape[1])
        datamodule = TimeseriesDataModule(
            SEQ_LEN, self.x_train, val_data=self.x_val, batch_size=BATCH_SIZE
        )
        trainer = AutoencoderTrainer(max_epochs=5, enable_progress_bar=True, limit_val_batches=1)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.x_test, SEQ_LEN), batch_size=1)
        y_test = trainer.predict(model, dataloaders=streamloader, unbatch=True)
        self.assertTupleEqual(self.x_test.shape, y_test.size())

    def test_trainer_06_w_val(self):
        model = Conv1dAE(seq_len=SEQ_LEN, in_channels=self.x_train.shape[1], enc_channels=(4,))
        train_dataset = StreamingDataset(self.x_train, SEQ_LEN)
        val_dataset = StreamingDataset(self.x_val, SEQ_LEN)
        test_dataset = StreamingDataset(self.x_test, SEQ_LEN)

        trainer = AutoencoderTrainer(max_epochs=5, enable_progress_bar=True)
        trainer.fit(
            model,
            train_dataloaders=DataLoader(train_dataset, batch_size=BATCH_SIZE),
            val_dataloaders=DataLoader(val_dataset, batch_size=BATCH_SIZE),
        )

        y_train = trainer.predict(
            model, dataloaders=DataLoader(train_dataset, batch_size=BATCH_SIZE)
        )
        self.assertTupleEqual(self.x_train.shape, y_train.size())

        streamloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        y_test_batched = trainer.predict(model, dataloaders=streamloader, unbatch=False)
        self.assertTupleEqual((229, SEQ_LEN, self.x_test.shape[1]), y_test_batched.size())

    def test_trainer_07_wo_val(self):
        model = SparseConv1dAE(
            seq_len=SEQ_LEN, in_channels=self.x_train.shape[1], enc_channels=(2,)
        )
        train_dataset = StreamingDataset(self.x_train, SEQ_LEN)

        test_dataset = StreamingDataset(self.x_test, SEQ_LEN)

        trainer = AutoencoderTrainer(max_epochs=25, enable_progress_bar=True)
        trainer.fit(model, train_dataloaders=DataLoader(train_dataset, batch_size=BATCH_SIZE))

        y_train = trainer.predict(
            model, dataloaders=DataLoader(train_dataset, batch_size=BATCH_SIZE)
        )
        self.assertTupleEqual(self.x_train.shape, y_train.size())

        streamloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        y_test_batched = trainer.predict(model, dataloaders=streamloader, unbatch=False)
        self.assertTupleEqual((229, SEQ_LEN, self.x_test.shape[1]), y_test_batched.size())


if __name__ == "__main__":
    unittest.main()
