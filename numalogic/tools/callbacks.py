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


import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBarBase


_LOGGER = logging.getLogger(__name__)


class ProgressDetails(ProgressBarBase):
    r"""
    A lightweight training progress detail producer.

    Args:
         log_freq: Interval of epochs to log
    """

    def __init__(self, log_freq: int = 5):
        super().__init__()
        self._log_freq = log_freq
        self._enable = True

    def enable(self) -> None:
        self._enable = True

    def disable(self):
        self._enable = False

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        curr_epoch = trainer.current_epoch
        if curr_epoch % self._log_freq == 0:
            _LOGGER.info("epoch %s, loss: %s", curr_epoch, metrics["loss"])
