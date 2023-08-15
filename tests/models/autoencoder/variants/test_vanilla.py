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
from numalogic.models.autoencoder.variants.vanilla import VanillaAE, SparseVanillaAE
from numalogic.tools.exceptions import LayerSizeMismatchError

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
EPOCHS = 5
BATCH_SIZE = 64
SEQ_LEN = 12
LR = 0.001
torch.manual_seed(42)


class TESTVanillaAE(unittest.TestCase):
    X_train = None
    X_val = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(df[:-240])
        cls.X_val = scaler.transform(df[-240:])

    def test_vanilla(self):
        model = VanillaAE(seq_len=SEQ_LEN, n_features=self.X_train.shape[1], weight_decay=1e-3)
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(fast_dev_run=True, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer()
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader)
        self.assertTupleEqual(self.X_val.shape, test_reconerr.shape)

    def test_sparse_vanilla(self):
        model = SparseVanillaAE(
            seq_len=SEQ_LEN, n_features=self.X_train.shape[1], loss_fn="l1", optim_algo="adagrad"
        )
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(fast_dev_run=True, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.X_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = AutoencoderTrainer()
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader, unbatch=False)
        self.assertTupleEqual((229, SEQ_LEN, self.X_train.shape[1]), test_reconerr.size())

    def test_native_train(self):
        model = VanillaAE(
            SEQ_LEN,
            n_features=2,
            encoder_layersizes=[24, 16, 6],
            decoder_layersizes=[6, 16, 24],
            optim_algo="rmsprop",
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

                loss = criterion(decoded, _X_batch)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print(f"epoch : {epoch}, loss_mean : {loss.item():.7f}")

    def test_train_err_01(self):
        with self.assertRaises(LayerSizeMismatchError):
            VanillaAE(
                SEQ_LEN,
                n_features=2,
                encoder_layersizes=[24, 16, 8],
                decoder_layersizes=[6, 16, 24],
            )

    def test_train_err_02(self):
        model = VanillaAE(SEQ_LEN, n_features=2, optim_algo="random")
        datamodule = TimeseriesDataModule(SEQ_LEN, self.X_train, batch_size=BATCH_SIZE)
        trainer = AutoencoderTrainer(max_epochs=EPOCHS, enable_progress_bar=True)
        with self.assertRaises(NotImplementedError):
            trainer.fit(model, datamodule=datamodule)

    def test_train_err_03(self):
        with self.assertRaises(NotImplementedError):
            VanillaAE(SEQ_LEN, n_features=2, loss_fn="random")


if __name__ == "__main__":
    unittest.main()
