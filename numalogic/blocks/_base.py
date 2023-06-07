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
from abc import ABCMeta, abstractmethod
from typing import Generic, Union

import numpy.typing as npt

from numalogic.tools.types import artifact_t, state_dict_t


class Block(Generic[artifact_t], metaclass=ABCMeta):
    """
    Base class for all blocks.

    A block is a unit of computation that can be
    chained together to form a pipeline. A block can be stateful or stateless.

    A stateful block is one that has a state that can be updated by calling the
    block with new data. A stateless block is one that does not have a state and
    can be called with new data without any side effects.

    A block can be used as a callable. The call method is an alias for the run method.

    Args:
    ----
        artifact: The artifact that the block operates on.
        name: The name of the block
        stateful: Whether the block is stateful or not. (default: True)
    """

    __slots__ = ("_name", "_stateful", "_artifact")

    def __init__(self, artifact: artifact_t, name: str, stateful: bool = True):
        self._artifact = artifact
        self._name = name
        self._stateful = stateful

    @property
    def name(self) -> str:
        """The name of the block."""
        return self._name

    @property
    def stateful(self) -> bool:
        """Whether the block is stateful or not."""
        return self._stateful

    @property
    def artifact(self) -> artifact_t:
        """The artifact that the block operates on."""
        return self._artifact

    @property
    def artifact_state(self) -> Union[artifact_t, state_dict_t]:
        """
        The state of the artifact that needs to be serialized for saving.

        This needs to be overridden if something other than the artifact itself
        needs to be serialized, e.g. statedict, or a torchscript module.
        """
        return self._artifact

    @artifact_state.setter
    def artifact_state(self, state: Union[artifact_t, state_dict_t]) -> None:
        """
        The state of the artifact that needs to be deserialized for loading.

        This needs to be overridden if something other than the artifact itself
        needs to be deserialized, e.g. statedict, or a torchscript module.
        """
        self._artifact = state

    def __call__(self, *args, **kwargs) -> npt.NDArray[float]:
        """Alias for the run method."""
        return self.run(*args, **kwargs)

    @abstractmethod
    def fit(self, data: npt.NDArray[float], *args, **kwargs):
        """
        Train the block on the input data.

        Implement this method to train the block, using the block's artifact.

        Args:
        ----
            data: The input data to train the block on.
            *args: Additional arguments for the block.
            **kwargs: Additional keyword arguments for fitting the block.
        """
        pass

    @abstractmethod
    def run(self, stream: npt.NDArray[float], *args, **kwargs) -> npt.NDArray[float]:
        """
        Run inference on the block on the streaming input data.

        Implement this method to run inference on the block,
        using the block's artifact.

        Args:
        ----
            stream: The streaming input data.
            *args: Additional arguments for the block.
            **kwargs: Additional keyword arguments for the block.
        """
        pass


class StatelessBlock(Block, metaclass=ABCMeta):
    """
    Base class for all stateless blocks.

    A stateless block is one that does not have a state and
    can be called with new data without any side effects.
    """

    def __init__(self, artifact: artifact_t, name: str):
        super().__init__(artifact, name, stateful=False)

    def fit(self, data: npt.NDArray[float], *args, **kwargs) -> npt.NDArray[float]:
        """
        A no-op for stateless blocks.

        Args:
        ----
            data: The input data to train the block on.
            *args: Additional arguments for the block.
            **kwargs: Additional keyword arguments for fitting the block.
        """
        return self.run(data, *args, **kwargs)
