import os
import unittest

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.tools.data import TimeseriesDataModule, StreamingDataset
from numalogic.models.autoencoder.trainer import AutoencoderTrainer
from numalogic.models.autoencoder.variants import LSTMAE
from numalogic.models.autoencoder.variants.lstm import SparseLSTMAE

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
EPOCHS = 2
BATCH_SIZE = 64
SEQ_LEN = 12
LR = 0.001
ACCELERATOR = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


class TestLSTMAE(unittest.TestCase):
    X_train = None
    X_val = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(df[:-240])
        cls.X_val = scaler.transform(df[-240:])

    def test_lstm_ae(self):
        model = LSTMAE(seq_len=SEQ_LEN, no_features=2, embedding_dim=15, weight_decay=1e-3)
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(
            accelerator=ACCELERATOR, fast_dev_run=True, enable_progress_bar=True
        )
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer(accelerator=ACCELERATOR)
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader)
        self.assertTupleEqual(self.X_val.shape, test_reconerr.shape)

    def test_sparse_lstm_ae(self):
        model = SparseLSTMAE(seq_len=SEQ_LEN, no_features=2, embedding_dim=15, loss_fn="mse")
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(
            accelerator=ACCELERATOR, fast_dev_run=True, enable_progress_bar=True
        )
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer(accelerator=ACCELERATOR)
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader, unbatch=False)
        self.assertListEqual([229, SEQ_LEN, self.X_train.shape[1]], list(test_reconerr.size()))

    def test_native_train(self):
        model = LSTMAE(seq_len=SEQ_LEN, no_features=2, embedding_dim=15)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.HuberLoss(delta=0.5)

        dataset = StreamingDataset(self.X_train, seq_len=SEQ_LEN)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        model.train()
        loss = torch.Tensor([0.0])
        for epoch in range(1, EPOCHS + 1):
            for _X_batch in train_loader:
                optimizer.zero_grad()
                encoded, decoded = model(_X_batch)

                loss = criterion(decoded, _X_batch)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print(f"epoch : {epoch}, mean loss : {loss.item():.7f}")


if __name__ == "__main__":
    unittest.main()
