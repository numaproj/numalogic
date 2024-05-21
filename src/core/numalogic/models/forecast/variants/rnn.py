import torch
from torch import nn, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.distributions import Normal


class GRUForecaster(pl.LightningModule):
    """Multistep forecasting using GRU."""

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 32,
        n_features: int = 1,
        forecast_horizon: int = 15,
        num_layers: int = 1,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.horizon = forecast_horizon
        self.weight_decay = weight_decay
        self.gru_1 = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.gru_2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x: Tensor):
        o, h = self.gru_1(x)
        h = h[-1, :, :].unsqueeze(1).repeat(1, self.horizon, 1)
        out, _ = self.gru_2(F.relu(h))
        return self.fc(F.relu(out))

    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=self.weight_decay)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        out = self.forward(x)
        # print(out.shape, y.shape)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int | None) -> Tensor:
        x, y = batch
        return self.forward(x)


class GRUIntervalForecaster(pl.LightningModule):
    """GRU Forecaster with interval prediction."""

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 32,
        n_features: int = 1,
        forecast_horizon: int = 15,
        num_layers: int = 1,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.horizon = forecast_horizon
        self.weight_decay = weight_decay
        self.gru_1 = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )  # L = 100
        self.gru_2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )  # L = 15
        self.mu = nn.Linear(hidden_size, n_features)
        self.logvar = nn.Linear(hidden_size, n_features)

    def forward(self, x: Tensor):
        o, h = self.gru_1(x)
        h = h[-1, :, :].unsqueeze(1).repeat(1, self.horizon, 1)
        out, _ = self.gru_2(F.relu(h))
        mu = self.mu(out)
        logvar = F.softplus(self.logvar(out))
        return mu, logvar

    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=self.weight_decay)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        mu, logvar = self.forward(x)
        dist = Normal(mu, logvar.exp())
        out = dist.rsample()
        # print(out.shape, y.shape)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int | None) -> Tensor:
        x, y = batch
        return self.forward(x)
