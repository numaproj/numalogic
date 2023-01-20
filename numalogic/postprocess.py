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
from numpy.typing import ArrayLike

from numalogic.tools import DataIndependentTransformers


def tanh_norm(scores: ArrayLike, scale_factor=10, smooth_factor=10) -> ArrayLike:
    return scale_factor * np.tanh(scores / smooth_factor)


class TanhNorm(DataIndependentTransformers):
    def __init__(self, scale_factor=10, smooth_factor=10):
        self.scale_factor = scale_factor
        self.smooth_factor = smooth_factor

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        return tanh_norm(X, scale_factor=self.scale_factor, smooth_factor=self.smooth_factor)
