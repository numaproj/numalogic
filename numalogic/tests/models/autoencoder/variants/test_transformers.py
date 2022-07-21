import os
import unittest

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.models.autoencoder.variants import TransformerAE
from numalogic.preprocess.datasets import SequenceDataset

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
EPOCHS = 10
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

    def test_train(self):
        model = TransformerAE(
            num_heads=8,
            seq_length=SEQ_LEN,
            dim_feedforward=64,
            num_encoder_layers=3,
            num_decoder_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.HuberLoss(delta=0.5)

        dataset = SequenceDataset(self.X_train, SEQ_LEN, permute=True)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model.summary(dataset.data.size())

        model.train()
        loss = torch.Tensor([0.0])
        for epoch in range(1, EPOCHS + 1):
            for _X_batch in train_loader:
                optimizer.zero_grad()
                encoded, decoded = model(_X_batch)

                loss = criterion(decoded, _X_batch)
                print(loss)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print(f"epoch : {epoch}, mean loss : {loss.item():.7f}")


if __name__ == "__main__":
    unittest.main()
