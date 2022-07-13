import os
import unittest

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.models.autoencoder.variants.vanilla import VanillaAE
from numalogic.preprocess.datasets import SequenceDataset
from numalogic.tools.exceptions import LayerSizeMismatchError

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
EPOCHS = 5
BATCH_SIZE = 256
SEQ_LEN = 20
LR = 0.001
torch.manual_seed(42)


class TESTBVanillaAE(unittest.TestCase):
    X_train = None
    X_val = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(df[:-240])
        cls.X_val = scaler.transform(df[-240:])

    def test_train_01(self):
        model = VanillaAE(SEQ_LEN, n_features=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.HuberLoss(delta=0.5)
        dataset = SequenceDataset(self.X_train, SEQ_LEN)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    def test_train_02(self):
        model = VanillaAE(
            SEQ_LEN, n_features=2, encoder_layersizes=[24, 16, 6], decoder_layersizes=[6, 16, 24]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.HuberLoss(delta=0.5)
        dataset = SequenceDataset(self.X_train, SEQ_LEN)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    def test_train_err(self):
        with self.assertRaises(LayerSizeMismatchError):
            VanillaAE(
                SEQ_LEN,
                n_features=2,
                encoder_layersizes=[24, 16, 8],
                decoder_layersizes=[6, 16, 24],
            )


if __name__ == "__main__":
    unittest.main()
