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

from collections.abc import Sequence
from collections.abc import Iterator

import numpy.typing as npt

from numalogic.blocks._transform import Block
from numalogic.registry import ArtifactManager
from numalogic.tools.types import artifact_t
from typing import Optional


class BlockPipeline(Sequence[Block]):
    """
    A pipeline of blocks.

    A pipeline is a sequence of blocks that can be chained together to form a
    pipeline. A pipeline can be used as a callable. The call method is an alias
    for the run method.

    Args:
    ----
        blocks: A list/tuple of blocks that form the pipeline.
        registry: The registry to use for storing artifacts.
    """

    __slots__ = ("_blocks", "_registry")

    def __init__(self, *blocks: Block, registry: Optional[ArtifactManager] = None):
        self._blocks = blocks
        self._registry = registry

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __getitem__(self, idx: int) -> Block:
        """Get the block at the given index."""
        return self._blocks[idx]

    def __len__(self) -> int:
        """Get the number of blocks in the pipeline."""
        return len(self._blocks)

    def __iter__(self) -> Iterator[Block]:
        """Get an iterator over the blocks in the pipeline."""
        return iter(self._blocks)

    def named_blocks(self) -> Iterator[tuple[str, Block]]:
        """Get an iterator over the blocks in the pipeline along with their names."""
        names = [block.name for block in self._blocks]
        return zip(names, self._blocks)

    def _get_block_params(self, **fit_params) -> dict[str, dict]:
        """
        Get the parameters for each block from the fit_params.

        Inspired by sklearn Pipeline
        (https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html).

        Args:
        ----
            fit_params : keyword dict of string -> object

        Returns
        -------
            A nested dict of blockname -> parameter -> value

        Raises
        ------
            ValueError: If the keyword arguments are not of the form
            blockname__parameter, e.g. `block_pipeline.fit(data, nn__max_epochs=50)`
        """
        block_params = {name: {} for name, block in self.named_blocks()}
        err_msg = (
            "Invalid kwarg: {pname} found. Keyword args of "
            "BlockPipeline must be of the form blockname__parameter, "
            "e.g. `block_pipeline.fit(data, nn__max_epochs=50)`"
        )
        for pname, pval in fit_params.items():
            if "__" not in pname:
                raise ValueError(err_msg.format(pname=pname))
            blockname, param = pname.split("__", 1)
            block_params[blockname][param] = pval
        return block_params

    def fit(self, input_: npt.NDArray[float], **fit_params) -> npt.NDArray[float]:
        """
        Fit the pipeline on the input data.

        Args:
        ----
            input_: The input data to fit the pipeline on.
            fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each block, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
            Final fit block output.
        """
        fit_params = self._get_block_params(**fit_params)
        for block in self._blocks:
            input_ = block.fit(input_, **fit_params.get(block.name, {}))
        return input_

    def run(self, data: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Perform inference on streaming data.

        Args:
        ----
            data: Streaming input data

        """
        for block in self._blocks:
            data = block.run(data)
        return data

    def save(self, skeys: Sequence[str], dkeys: Sequence[str]) -> None:
        """
        Save the state of the pipeline.

        Args:
        ----
            skeys: Sequence of source keys.
            dkeys: Sequence of destination keys.

        Raises
        ------
            ValueError: If no registry is provided.
        """
        if not self._registry:
            raise ValueError("No registry provided.")

        artifacts: dict[str, artifact_t] = {}
        for block in self._blocks:
            if not block.stateful:
                continue
            artifacts[block.name] = block.artifact_state
        self._registry.save(skeys, dkeys, artifacts)

    def load(self, skeys: Sequence[str], dkeys: Sequence[str]) -> None:
        """
        Load the state of the pipeline.

        Args:
        ----
            skeys: Sequence of source keys.
            dkeys: Sequence of destination keys.

        Raises
        ------
            ValueError: If no registry is provided.
        """
        if not self._registry:
            raise ValueError("No registry provided.")

        artifact_data = self._registry.load(skeys, dkeys)
        artifacts = artifact_data.artifact
        for block in self._blocks:
            if not block.stateful:
                continue
            block.artifact_state = artifacts[block.name]
