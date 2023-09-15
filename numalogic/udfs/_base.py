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
from typing import Union
from collections.abc import Coroutine
import numpy.typing as npt
from pynumaflow.function import Datum, Messages

from numalogic.tools.types import artifact_t


class NumalogicUDF(metaclass=ABCMeta):
    """
    Base class for all Numalogic based UDFs.

    Args:
        is_async: If True, the UDF is executed in an asynchronous manner.
    """

    __slots__ = ("is_async",)

    def __init__(self, is_async: bool = False):
        self.is_async = is_async

    def __call__(
        self, keys: list[str], datum: Datum
    ) -> Union[Coroutine[None, None, Messages], Messages]:
        if self.is_async:
            return self.aexec(keys, datum)
        return self.exec(keys, datum)

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        Called when the UDF is executed in a synchronous manner.

        Args:
            keys: list of keys.
            datum: Datum object.

        Returns
        -------
            Messages instance
        """
        raise NotImplementedError("exec method not implemented")

    async def aexec(self, keys: list[str], datum: Datum) -> Messages:
        """
        Called when the UDF is executed in an asynchronous manner.

        Args:
            keys: list of keys.
            datum: Datum object.

        Returns
        -------
            Messages instance
        """
        raise NotImplementedError("aexec method not implemented")

    @classmethod
    @abstractmethod
    def compute(cls, model: artifact_t, input_: npt.NDArray[float], **kwargs):
        """
        Abstract method to be implemented by subclasses.

        Args:
            model: artifact for the udf.
            input_: Input array.
            kwargs: Additional keyword arguments.
        """
        pass
