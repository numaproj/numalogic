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
EPOCHS = 5
BATCH_SIZE = 256
SEQ_LEN = 12
LR = 0.001
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

    def test_conv1d(self):
        model = Conv1dAE(seq_len=SEQ_LEN, in_channels=self.X_train.shape[1], enc_channels=8)
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(max_epochs=5, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer()
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader)
        self.assertTupleEqual(self.X_val.shape, test_reconerr.shape)

    def test_sparse_conv1d(self):
        model = SparseConv1dAE(
            seq_len=SEQ_LEN, in_channels=self.X_train.shape[1], enc_channels=8, loss_fn="mse"
        )
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(max_epochs=5, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer()
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader, unbatch=False)
        self.assertTupleEqual((229, SEQ_LEN, self.X_train.shape[1]), test_reconerr.size())

    def test_native_train(self):
        model = Conv1dAE(seq_len=SEQ_LEN, in_channels=self.X_train.shape[1], enc_channels=8)
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
                decoded = decoded.view(-1, SEQ_LEN, self.X_train.shape[1])

                loss = criterion(decoded, _X_batch)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print(f"epoch : {epoch}, loss_mean : {loss.item():.7f}")


if __name__ == "__main__":
    unittest.main()
