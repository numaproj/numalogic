import logging
import os
import unittest

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.models.vae import VAETrainer
from numalogic.models.vae.variants import Conv1dVAE
from numalogic.tools.data import TimeseriesDataModule, StreamingDataset
from numalogic.tools.exceptions import ModelInitializationError

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
EPOCHS = 2
BATCH_SIZE = 32
SEQ_LEN = 12
LR = 0.001
ACCELERATOR = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


logging.basicConfig(level=logging.INFO)


class TestConv1dVAE(unittest.TestCase):
    x_train = None
    x_val = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]
        scaler = StandardScaler()
        cls.x_train = scaler.fit_transform(df[:-240])
        cls.x_val = scaler.transform(df[-240:])

    def test_model_01(self):
        model = Conv1dVAE(seq_len=SEQ_LEN, n_features=2, latent_dim=1, loss_fn="l1")
        datamodule = TimeseriesDataModule(SEQ_LEN, self.x_train, batch_size=BATCH_SIZE)
        trainer = VAETrainer(accelerator=ACCELERATOR, max_epochs=EPOCHS, fast_dev_run=True)
        trainer.fit(model, datamodule=datamodule)

        streamloader = DataLoader(StreamingDataset(self.x_val, SEQ_LEN), batch_size=BATCH_SIZE)
        stream_trainer = VAETrainer(accelerator=ACCELERATOR)
        test_reconerr = stream_trainer.predict(model, dataloaders=streamloader)
        test_reconerr_w_seq = stream_trainer.predict(model, dataloaders=streamloader, unbatch=False)

        self.assertTupleEqual(self.x_val.shape, test_reconerr.shape)
        self.assertTupleEqual(streamloader.dataset.as_tensor().shape, test_reconerr_w_seq.shape)

    def test_model_02(self):
        model = Conv1dVAE(seq_len=SEQ_LEN, n_features=2, latent_dim=1, conv_channels=(8, 4))
        trainer = VAETrainer(accelerator=ACCELERATOR, max_epochs=EPOCHS, log_freq=1)
        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                StreamingDataset(self.x_train, SEQ_LEN), batch_size=BATCH_SIZE
            ),
        )

        test_ds = StreamingDataset(self.x_val, SEQ_LEN)

        model.eval()
        with torch.no_grad():
            _, recon = model(test_ds.as_tensor())

        self.assertTupleEqual(test_ds.as_tensor().size(), recon.shape)
        self.assertEqual(recon.dim(), 3)

    def test_native_train(self):
        model = Conv1dVAE(
            seq_len=SEQ_LEN,
            n_features=2,
            latent_dim=1,
            loss_fn="huber",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.HuberLoss(delta=0.5)

        train_loader = DataLoader(
            StreamingDataset(self.x_train, seq_len=SEQ_LEN), batch_size=BATCH_SIZE
        )

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

    def test_err(self):
        with self.assertRaises(ValueError):
            Conv1dVAE(
                seq_len=SEQ_LEN,
                n_features=2,
                latent_dim=1,
                loss_fn="random",
            )
        with self.assertRaises(ModelInitializationError):
            Conv1dVAE(
                seq_len=SEQ_LEN,
                n_features=2,
                latent_dim=1,
                conv_channels=(8, 4, 2, 1),
            )


if __name__ == "__main__":
    unittest.main()
