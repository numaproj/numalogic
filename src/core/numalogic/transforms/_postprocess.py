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

import numpy as np
import numpy.typing as npt

from numalogic.base import StatelessTransformer


def tanh_norm(scores: npt.NDArray[float], scale_factor=10, smooth_factor=10) -> npt.NDArray[float]:
    """
    Applies column wise tanh normalization to the input data.

    This is most commonly used to normalize the anomaly scores to a desired range.

    Args:
    ----
        scores: feature array
        scale_factor: scale the output by this factor (default: 10)
        smooth_factor: factor to broaden out the linear range of the graph (default: 10)
    """
    return scale_factor * np.tanh(scores / smooth_factor)


class TanhNorm(StatelessTransformer):
    """
    Apply tanh normalization to the input data.

    Args:
    ----
        scale_factor: scale the output by this factor (default: 10)
        smooth_factor: factor to broaden out the linear range of the graph (default: 10)
    """

    __slots__ = ("scale_factor", "smooth_factor")

    def __init__(self, scale_factor=10, smooth_factor=10):
        self.scale_factor = scale_factor
        self.smooth_factor = smooth_factor

    def transform(self, input_: npt.NDArray[float], **__) -> npt.NDArray[float]:
        return tanh_norm(input_, scale_factor=self.scale_factor, smooth_factor=self.smooth_factor)
