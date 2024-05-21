from typing import Callable

import torch.nn.functional as F
from torch import Tensor, optim

from numalogic.base import TorchModel


def _init_criterion(loss_fn: str) -> Callable:
    if loss_fn == "huber":
        return F.huber_loss
    if loss_fn == "l1":
        return F.l1_loss
    if loss_fn == "mse":
        return F.mse_loss
    raise ValueError(f"Unsupported loss function provided: {loss_fn}")


class BaseVAE(TorchModel):
    """
    Abstract Base class for all Pytorch based variational autoencoder models.

    Args:
    ----
        lr: learning rate (default: 3e-4)
        weight_decay: weight decay factor weight for regularization (default: 0.0)
        loss_fn: loss function used to train the model
                    supported values include: {mse, l1, huber}
    """

    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        loss_fn: str = "mse",
    ):
        super().__init__()
        self._lr = lr
        self.weight_decay = weight_decay
        self.criterion = _init_criterion(loss_fn)

    def configure_shape(self, x: Tensor) -> Tensor:
        """Method to configure the batch shape for each type of model architecture."""
        return x

    def configure_optimizers(self) -> dict:
        optimizer = optim.Adam(self.parameters(), lr=self._lr, weight_decay=self.weight_decay)
        return {"optimizer": optimizer}

    def recon_loss(self, batch: Tensor, recon: Tensor, reduction: str = "sum"):
        return self.criterion(batch, recon, reduction=reduction)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Validation step for the model."""
        p, recon = self.forward(batch)
        loss = self.recon_loss(batch, recon)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Prediction step for the model."""
        p, recon = self.forward(batch)
        return self.recon_loss(batch, recon, reduction="none")
