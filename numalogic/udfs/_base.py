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
from typing import Union, Optional
from collections.abc import Coroutine
import numpy.typing as npt
from pynumaflow.mapper import Datum, Messages

from numalogic.tools.exceptions import ConfigNotFoundError
from numalogic.tools.types import artifact_t
from numalogic.udfs._config import PipelineConf, StreamConf


_DEFAULT_CONF_ID = "default"


class NumalogicUDF(metaclass=ABCMeta):
    """
    Base class for all Numalogic based UDFs.

    Args:
        is_async: If True, the UDF is executed in an asynchronous manner.
        pl_conf: PipelineConf object
    """

    __slots__ = ("is_async", "pl_conf")

    def __init__(self, is_async: bool = False, pl_conf: Optional[PipelineConf] = None):
        self.is_async = is_async
        self.pl_conf = pl_conf or PipelineConf()

    def __call__(
        self, keys: list[str], datum: Datum
    ) -> Union[Coroutine[None, None, Messages], Messages]:
        if self.is_async:
            return self.aexec(keys, datum)
        return self.exec(keys, datum)

    # TODO: remove, and have an update config method
    def register_conf(self, config_id: str, conf: StreamConf) -> None:
        """
        Register config with the UDF.

        Args:
            config_id: Config ID
            conf: StreamConf object
        """
        self.pl_conf.stream_confs[config_id] = conf

    def _get_default_conf(self, config_id) -> StreamConf:
        """Get the default config."""
        try:
            return self.pl_conf.stream_confs[_DEFAULT_CONF_ID]
        except KeyError:
            err_msg = f"Config with ID {config_id} or {_DEFAULT_CONF_ID} not found!"
            raise ConfigNotFoundError(err_msg) from None

    def get_conf(self, config_id: str) -> StreamConf:
        """
        Get config with the given ID.
        If not found, return the default config.

        Args:
            config_id: Config ID

        Returns
        -------
            StreamConf object

        Raises
        ------
            ConfigNotFoundError: If config with the given ID is not found
        """
        try:
            return self.pl_conf.stream_confs[config_id]
        except KeyError:
            return self._get_default_conf(config_id)

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
