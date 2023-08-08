import sys
import warnings
from typing import Optional

import torch
from torch import Tensor
from pytorch_lightning import Trainer, LightningModule

from numalogic.tools.callbacks import ConsoleLogger
from numalogic.tools.data import inverse_window


class VAETrainer(Trainer):
    """A PyTorch Lightning Trainer for VAE models.

    Args:
    ----
        max_epochs: The maximum number of epochs to train for. (default: 100)
        logger: Whether to use a console logger to log metrics. (default: True)
        log_freq: The number of epochs between logging. (default: 5)
        check_val_every_n_epoch: The number of epochs between validation checks. (default: 5)
        enable_checkpointing: Whether to enable checkpointing. (default: False)
        enable_progress_bar: Whether to enable the progress bar. (default: False)
        enable_model_summary: Whether to enable the model summary. (default: False)
        **trainer_kw: Additional keyword arguments to pass to the Lightning Trainer.
    """

    def __init__(
        self,
        max_epochs: int = 100,
        logger: bool = True,
        log_freq: int = 5,
        check_val_every_n_epoch: int = 5,
        enable_checkpointing: bool = False,
        enable_progress_bar: bool = False,
        enable_model_summary: bool = False,
        **trainer_kw
    ):
        if not sys.warnoptions:
            warnings.simplefilter("ignore", category=UserWarning)

        if logger:
            logger = ConsoleLogger(log_freq=log_freq)

        super().__init__(
            logger=logger,
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            **trainer_kw
        )

    def predict(self, model: Optional[LightningModule] = None, unbatch=True, **kwargs) -> Tensor:
        r"""Predicts the output of the model.

        Args:
        ----
            model: The model to predict with. (default: None)
            unbatch: Whether to inverse window the output. (default: True)
            **kwargs: Additional keyword arguments to pass to the Lightning
                      trainers predict method.
        """
        recon_err = super().predict(model, **kwargs)
        recon_err = torch.vstack(recon_err)
        if unbatch:
            return inverse_window(recon_err, method="keep_last")
        return recon_err
