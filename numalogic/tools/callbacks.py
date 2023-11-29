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
from typing import Optional, Union

import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.loggers import Logger

_LOGGER = logging.getLogger(__name__)


class ProgressDetails(ProgressBar):
    r"""A lightweight training progress detail producer.

    Args:
    ----
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
        loss = pl_module.total_train_loss / trainer.num_training_batches
        curr_epoch = trainer.current_epoch
        if curr_epoch % self._log_freq == 0:
            _LOGGER.info("epoch=%s, training_loss=%.5f", curr_epoch, loss)
        pl_module.reset_train_loss()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        loss = pl_module.total_val_loss / trainer.num_val_batches[0]
        _LOGGER.info("validation_loss=%.5f", loss)
        pl_module.reset_val_loss()


class ConsoleLogger(Logger):
    """
    A lightweight console logger for training metrics.

    Args:
    ----
        log_freq: Interval of epochs to log
        experiment_id: Optional experiment id, e.g. a uuid
        experiment_name: Optional experiment name, e.g. a model name
    """

    def __init__(
        self,
        log_freq: int = 5,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        self.log_freq = log_freq
        self._id = experiment_id
        self._name = experiment_name

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self._id

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def experiment(self) -> Optional[str]:
        return self._name

    def log_hyperparams(self, params, *args, **kwargs):
        raise NotImplementedError("ConsoleLogger does not log hyperparameters")

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        epoch = metrics.pop("epoch", None)
        if epoch is None:
            _LOGGER.info("metrics=%s", metrics)
            return
        epoch += 1
        if epoch == 1:
            _LOGGER.info(self._format_metrics(epoch, metrics))
        elif not (epoch % self.log_freq):
            _LOGGER.info(self._format_metrics(epoch, metrics))

    @staticmethod
    def _format_metrics(epoch, metrics: dict[str, float]) -> str:
        log_msg = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        return f"epoch={epoch}, {log_msg}"
