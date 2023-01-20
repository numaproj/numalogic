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
import torch
from pytorch_lightning import Trainer
from torch import Tensor

from numalogic.tools.callbacks import ProgressDetails
from numalogic.tools.data import TimeseriesDataModule

_LOGGER = logging.getLogger(__name__)


class AutoencoderTrainer(Trainer):
    def __init__(
        self,
        max_epochs=100,
        logger=False,
        check_val_every_n_epoch=5,
        log_every_n_steps=20,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        callbacks=None,
        **trainer_kw
    ):
        if (not callbacks) and enable_progress_bar:
            callbacks = ProgressDetails()

        super().__init__(
            logger=logger,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            limit_val_batches=limit_val_batches,
            callbacks=callbacks,
            **trainer_kw
        )

    def predict(self, model: pl.LightningModule = None, unbatch=True, **kwargs) -> Tensor:
        recon_err = super().predict(model, **kwargs)
        recon_err = torch.vstack(recon_err)
        if unbatch:
            return TimeseriesDataModule.unbatch_sequences(recon_err)
        return recon_err
