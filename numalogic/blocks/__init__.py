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

"""
Module for numalogic blocks which are units of computation that can be
chained together to form a pipeline if needed. A block can be stateful or stateless.
"""

from numalogic.blocks._base import Block
from numalogic.blocks._nn import NNBlock
from numalogic.blocks._transform import PreprocessBlock, PostprocessBlock, ThresholdBlock
from numalogic.blocks.pipeline import BlockPipeline

__all__ = [
    "Block",
    "NNBlock",
    "PreprocessBlock",
    "PostprocessBlock",
    "ThresholdBlock",
    "BlockPipeline",
]
