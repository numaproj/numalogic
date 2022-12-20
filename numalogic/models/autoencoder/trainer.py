import logging

import pytorch_lightning as pl
import torch
from numalogic.tools.data import TimeseriesDataModule
from pytorch_lightning import Trainer
from torch import Tensor

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
        **trainer_kw
    ):
        super().__init__(
            logger=logger,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            limit_val_batches=limit_val_batches,
            **trainer_kw
        )

    def predict(self, model: pl.LightningModule = None, unbatch=True, **kwargs) -> Tensor:
        recon_err = super().predict(model, **kwargs)
        recon_err = torch.vstack(recon_err)
        if unbatch:
            return TimeseriesDataModule.unbatch_sequences(recon_err)
        return recon_err
