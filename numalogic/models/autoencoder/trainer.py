import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch import Tensor

_LOGGER = logging.getLogger(__name__)


class AutoencoderTrainer(Trainer):
    def __init__(
        self,
        max_epochs=100,
        logger=False,
        check_val_every_n_epoch=10,
        log_every_n_steps=20,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
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
            **trainer_kw
        )

    def predict(self, model: pl.LightningModule = None, **kwargs) -> Tensor:
        outs = super().predict(model, **kwargs)
        return torch.vstack(outs)
