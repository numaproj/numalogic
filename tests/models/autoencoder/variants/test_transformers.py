import os
import unittest

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.tools.data import StreamingDataset, TimeseriesDataModule
from numalogic.models.autoencoder.trainer import AutoencoderTrainer
from numalogic.models.autoencoder.variants import TransformerAE
from numalogic.models.autoencoder.variants.transformer import SparseTransformerAE

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
EPOCHS = 2
BATCH_SIZE = 256
SEQ_LEN = 12
LR = 0.001
torch.manual_seed(42)


class TestTransformerAE(unittest.TestCase):
    X_train = None
    X_val = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(df[:-240])
        cls.X_val = scaler.transform(df[-240:])

    def test_transformer(self):
        model = TransformerAE(
            seq_len=SEQ_LEN,
            n_features=2,
            num_heads=8,
            dim_feedforward=64,
            num_encoder_layers=3,
            num_decoder_layers=1,
        )
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(accelerator="cpu", fast_dev_run=True, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer(accelerator="cpu")
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader)
        self.assertTupleEqual(self.X_val.shape, test_reconerr.shape)

    def test_sparse_transformer(self):
        model = SparseTransformerAE(seq_len=SEQ_LEN, n_features=self.X_train.shape[1], loss_fn="l1")
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(accelerator="cpu", fast_dev_run=True, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer(accelerator="cpu")
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader, unbatch=False)
        self.assertListEqual([229, SEQ_LEN, self.X_train.shape[1]], list(test_reconerr.size()))

    def test_train(self):
        model = TransformerAE(
            n_features=2,
            num_heads=8,
            seq_len=SEQ_LEN,
            dim_feedforward=64,
            num_encoder_layers=3,
            num_decoder_layers=1,
            weight_decay=1e-3,
        )
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
                decoded = decoded.view(-1, SEQ_LEN, self.X_train.shape[1])

                loss = criterion(decoded, _X_batch)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print(f"epoch : {epoch}, mean loss : {loss.item():.7f}")


if __name__ == "__main__":
    unittest.main()
