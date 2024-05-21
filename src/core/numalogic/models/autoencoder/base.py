# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any

from torch import Tensor, optim

from numalogic.base import TorchModel
from numalogic.tools.loss import get_loss_fn


class BaseAE(TorchModel):
    r"""Abstract Base class for all Pytorch based autoencoder models for time-series data.

    Args:
    ----
        loss_fn: loss function used to train the model
                 supported values include: {huber, l1, mae}
        optim_algo: optimizer algo to be used for training
                    supported values include: {adam, adagrad, rmsprop}
        lr: learning rate (default: 1e-3)
        weight_decay: weight decay factor weight for regularization (default: 0.0)
    """

    def __init__(
        self,
        loss_fn: str = "huber",
        optim_algo: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.lr = lr
        self.optim_algo = optim_algo
        self.criterion = get_loss_fn(loss_fn)
        self.weight_decay = weight_decay

    def init_optimizer(self, optim_algo: str):
        if optim_algo == "adam":
            return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if optim_algo == "adagrad":
            return optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if optim_algo == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raise NotImplementedError(f"Unsupported optimizer value provided: {optim_algo}")

    def configure_shape(self, x: Tensor) -> Tensor:
        """Method to configure the batch shape for each type of model architecture."""
        return x

    def get_reconstruction_loss(self, batch: Tensor, reduction="mean") -> Tensor:
        _, recon = self.forward(batch)
        return self.criterion(batch, recon, reduction=reduction)

    def reconstruction(self, batch: Tensor) -> Tensor:
        _, recon = self.forward(batch)
        return recon

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.init_optimizer(self.optim_algo)
        return {"optimizer": optimizer}

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        recon_loss = self.get_reconstruction_loss(batch)
        self.log("train_loss", recon_loss, on_epoch=True, on_step=False)
        return recon_loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        recon_loss = self.get_reconstruction_loss(batch)
        self.log("val_loss", recon_loss)
        return recon_loss
