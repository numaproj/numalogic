import os
import unittest

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.tools.data import TimeseriesDataModule, StreamingDataset
from numalogic.models.autoencoder.trainer import AutoencoderTrainer
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.models.autoencoder.variants.conv import SparseConv1dAE

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
EPOCHS = 2
BATCH_SIZE = 32
SEQ_LEN = 12
LR = 0.001
ACCELERATOR = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


class TestConvAE(unittest.TestCase):
    X_train = None
    X_val = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(df[:-240])
        cls.X_val = scaler.transform(df[-240:])

    def test_conv1d_1(self):
        model = Conv1dAE(seq_len=SEQ_LEN, in_channels=self.X_train.shape[1], enc_channels=[8])
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(max_epochs=2, enable_progress_bar=True, fast_dev_run=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer()
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader)
        self.assertTupleEqual(self.X_val.shape, test_reconerr.shape)

    def test_conv1d_2(self):
        model = Conv1dAE(
            seq_len=SEQ_LEN,
            in_channels=self.X_train.shape[1],
            enc_channels=[8, 16, 4],
            enc_kernel_sizes=[3, 3, 3],
            dec_activation="sigmoid",
            weight_decay=1e-3,
            optim_algo="rmsprop",
        )
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(
            accelerator=ACCELERATOR, enable_progress_bar=True, fast_dev_run=True
        )
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer(accelerator=ACCELERATOR)
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader)
        self.assertTupleEqual(self.X_val.shape, test_reconerr.shape)

    def test_conv1d_encode(self):
        model = Conv1dAE(
            seq_len=SEQ_LEN,
            in_channels=self.X_train.shape[1],
            enc_channels=[8, 16, 4],
            enc_kernel_sizes=[3, 3, 3],
            dec_activation="sigmoid",
        )
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(accelerator=ACCELERATOR, fast_dev_run=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=256)
        for batch in streamloader:
            latent = model.encode(batch)
            self.assertTupleEqual((229, 4, 3), latent.shape)
            break

    def test_conv1d_err(self):
        with self.assertRaises(AssertionError):
            Conv1dAE(
                seq_len=SEQ_LEN,
                in_channels=self.X_train.shape[1],
                enc_channels=[8, 16, 4],
                enc_kernel_sizes=[3, 3],
            )

        with self.assertRaises(ValueError):
            Conv1dAE(
                seq_len=SEQ_LEN,
                in_channels=self.X_train.shape[1],
                enc_channels=[8, 16, 4],
                enc_kernel_sizes=[3, 3, 3],
                dec_activation="random",
            )

        with self.assertRaises(TypeError):
            Conv1dAE(
                seq_len=SEQ_LEN,
                in_channels=self.X_train.shape[1],
                enc_channels=[8, 16, 4],
                enc_kernel_sizes={5, 3, 1},
                dec_activation="random",
            )

    def test_sparse_conv1d(self):
        model = SparseConv1dAE(
            seq_len=SEQ_LEN,
            in_channels=self.X_train.shape[1],
            enc_channels=(16, 4),
            enc_kernel_sizes=(5, 3),
            loss_fn="mse",
            dec_activation="relu",
        )
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(
            accelerator=ACCELERATOR, enable_progress_bar=True, fast_dev_run=True
        )
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer(accelerator=ACCELERATOR)
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader, unbatch=False)
        self.assertTupleEqual((229, SEQ_LEN, self.X_train.shape[1]), test_reconerr.size())

    def test_native_train(self):
        model = Conv1dAE(
            seq_len=SEQ_LEN,
            in_channels=self.X_train.shape[1],
            enc_channels=[8, 16, 4],
            enc_kernel_sizes=(3, 3, 5),
            dec_activation="tanh",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.HuberLoss(delta=0.5)

        dataset = StreamingDataset(self.X_train, seq_len=SEQ_LEN)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        model.train()
        loss = Tensor([0.0])
        for epoch in range(1, EPOCHS + 1):
            for _X_batch in train_loader:
                optimizer.zero_grad()
                encoded, decoded = model(_X_batch)

                loss = criterion(decoded, _X_batch)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print(f"epoch : {epoch}, loss_mean : {loss.item():.7f}")


if __name__ == "__main__":
    unittest.main()
