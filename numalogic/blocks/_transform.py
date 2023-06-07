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

import numpy.typing as npt

from numalogic.blocks._base import Block, StatelessBlock
from numalogic.tools.types import transform_t, thresh_t


class PreprocessBlock(Block):
    """
    A stateful block that is used to preprocess the input data, before it is fed to an ML model.

    Serialization is done by saving the preprocessor object.

    Args:
    ----
        preprocessor: The preprocessor object.
        name: The name of the block. Defaults to "preprocess".
        stateful: Whether the block is stateful or not. Defaults to True.
    """

    def __init__(self, preprocessor: transform_t, name: str = "preprocess", stateful: bool = True):
        super().__init__(artifact=preprocessor, name=name, stateful=stateful)

    def fit(self, input_: npt.NDArray[float], **__) -> npt.NDArray[float]:
        """
        Fit the preprocessor on the input data.

        Args:
        ----
            input_: The input data to train on.

        Returns
        -------
            The transformed/scaled input data.
        """
        return self._artifact.fit_transform(input_)

    def run(self, input_: npt.NDArray[float], **__) -> npt.NDArray[float]:
        """
        Transform the streaming input data.

        Args:
        ----
            input_: The streaming input data.

        Returns
        -------
            The transformed/scaled input data.
        """
        return self._artifact.transform(input_)


class ThresholdBlock(Block):
    """
    A stateful block that is used to threshold the output of an ML model.

    Serialization is done by saving the threshold object.

    Args:
    ----
        thresh_model: The threshold model object.
        name: The name of the block. Defaults to "threshold".
    """

    def __init__(self, thresh_model: thresh_t, name: str = "threshold"):
        super().__init__(artifact=thresh_model, name=name)

    def fit(self, input_: npt.NDArray[float], **__) -> npt.NDArray[float]:
        """
        Fit the threshold model on the training data.

        Args:
        ----
            input_: The input data to train on.

        Returns
        -------
            The anomaly scores of the training data.
        """
        self._artifact.fit(input_)
        return self._artifact.score_samples(input_)

    def run(self, input_: npt.NDArray[float], **__) -> npt.NDArray[float]:
        """
        Transform the streaming input data.

        Args:
        ----
            input_: The streaming input data.

        Returns
        -------
            The anomaly score of the streaming data.
        """
        return self._artifact.score_samples(input_)


class PostprocessBlock(StatelessBlock):
    """
    A stateless block that is used to postprocess the output of an ML model.

    Args:
    ----
        postprocessor: The postprocessor object.
    """

    def __init__(self, postprocessor: transform_t, name: str = "postprocess"):
        super().__init__(artifact=postprocessor, name=name)

    def run(self, input_: npt.NDArray[float], **__) -> npt.NDArray[float]:
        """
        Transform the streaming input data.

        Args:
        ----
            input_: The streaming input data.

        Returns
        -------
            The postprocessed streaming data.
        """
        return self._artifact.transform(input_)
