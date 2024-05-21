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

"""Base classes for all models and transforms."""


from abc import ABCMeta

import numpy.typing as npt
import pytorch_lightning as pl
from sklearn.base import TransformerMixin, OutlierMixin


class BaseTransformer(TransformerMixin):
    """Base class for all transformer classes."""

    pass


class StatelessTransformer(BaseTransformer):
    """Base class for stateless transforms."""

    def transform(self, input_: npt.NDArray, **__):
        """Implement the transform method."""
        raise NotImplementedError("transform method not implemented")

    def fit(self, _: npt.NDArray):
        """Fit method does nothing for stateless transforms."""
        return self

    def fit_transform(self, input_: npt.NDArray, _=None, **__):
        """Return the result of the transform method."""
        return self.transform(input_)


class TorchModel(pl.LightningModule, metaclass=ABCMeta):
    """Base class for all Pytorch based models."""

    pass


class BaseThresholdModel(OutlierMixin):
    """Base class for all threshold models."""

    pass
