from abc import ABCMeta

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor, optim


class BaseAE(pl.LightningModule, metaclass=ABCMeta):
    """
    Abstract Base class for all Pytorch based autoencoder models for time-series data.
    """

    def __init__(self, loss_fn: str = "huber", optim_algo: str = "adam", lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.optim_algo = optim_algo
        self.criterion = self.init_criterion(loss_fn)

    @staticmethod
    def init_criterion(loss_fn: str):
        if loss_fn == "huber":
            return F.huber_loss
        if loss_fn == "l1":
            return F.l1_loss
        if loss_fn == "mse":
            return F.mse_loss
        raise NotImplementedError(f"Unsupported loss function provided: {loss_fn}")

    def init_optimizer(self, optim_algo: str):
        if optim_algo == "adam":
            return optim.Adam(self.parameters(), lr=self.lr)
        if optim_algo == "adagrad":
            return optim.Adagrad(self.parameters(), lr=self.lr)
        if optim_algo == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.lr)
        raise NotImplementedError(f"Unsupported optimizer value provided: {optim_algo}")

    def _get_reconstruction_loss(self, batch):
        _, recon = self.forward(batch)
        return self.criterion(batch, recon)

    def reconstruction(self, batch: Tensor) -> Tensor:
        _, recon = self.forward(batch)
        return recon

    def configure_optimizers(self):
        optimizer = self.init_optimizer(self.optim_algo)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        return loss
