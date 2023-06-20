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

import torch
from torch.utils.data import DataLoader
import numpy.typing as npt

from numalogic.blocks import Block
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.tools.data import StreamingDataset
from numalogic.tools.types import nn_model_t, state_dict_t


class NNBlock(Block):
    """
    A block that uses a neural network model to operate on the artifact.

    Serialization is done by saving state dict of the model.

    Args:
    ----
        model: The neural network model.
        seq_len: The sequence length of the input data.
        name: The name of the block. Defaults to "nn".
    """

    __slots__ = ("seq_len",)

    def __init__(self, model: nn_model_t, seq_len: int, name: str = "nn"):
        super().__init__(artifact=model, name=name)
        self.seq_len = seq_len

    @property
    def artifact_state(self) -> state_dict_t:
        """The state dict of the model."""
        return self._artifact.state_dict()

    @artifact_state.setter
    def artifact_state(self, artifact_state: state_dict_t) -> None:
        """Set the state dict of the model."""
        self._artifact.load_state_dict(artifact_state)

    def fit(
        self, input_: npt.NDArray[float], batch_size: int = 64, **trainer_kwargs
    ) -> npt.NDArray[float]:
        """
        Train the model on the input data.

        Args:
        ----
            input_: The input data.
            batch_size: The batch size to use for training.
            trainer_kwargs: Keyword arguments to pass to the lightning trainer.

        Returns
        -------
            The error of the model on the input data.
        """
        trainer = AutoencoderTrainer(**trainer_kwargs)
        ds = StreamingDataset(input_, self.seq_len)
        trainer.fit(self._artifact, train_dataloaders=DataLoader(ds, batch_size=batch_size))
        reconerr = trainer.predict(
            self._artifact, dataloaders=DataLoader(ds, batch_size=batch_size)
        )
        return reconerr.numpy()

    def run(self, input_: npt.NDArray[float], **_) -> npt.NDArray[float]:
        """
        Perform forward pass on the streaming input data.

        Args:
        ----
            input_: The streaming input data.

        Returns
        -------
            The error of the model on the input data.
        """
        input_ = torch.from_numpy(input_).float()
        # Add a batch dimension
        input_ = torch.unsqueeze(input_, dim=0).contiguous()
        self._artifact.eval()
        with torch.no_grad():
            reconerr = self._artifact.predict_step(input_, batch_idx=0)
        return torch.squeeze(reconerr, dim=0).numpy()
