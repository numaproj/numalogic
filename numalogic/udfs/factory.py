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

import logging
import warnings
from typing import ClassVar, TypeVar

_LOGGER = logging.getLogger(__name__)


class UDFFactory:
    """Factory class to fetch the right UDF."""

    from numalogic.udfs import NumalogicUDF
    from numalogic.udfs.payloadtx import PayloadTransformer
    from numalogic.udfs.inference import InferenceUDF
    from numalogic.udfs.postprocess import PostprocessUDF
    from numalogic.udfs.preprocess import PreprocessUDF
    from numalogic.udfs.trainer import DruidTrainerUDF, PromTrainerUDF, RDSTrainerUDF
    from numalogic.udfs.staticthresh import StaticThresholdUDF

    nl_udf_t = TypeVar("nl_udf_t", bound=NumalogicUDF, covariant=True)
    _UDF_MAP: ClassVar[dict[str, type[NumalogicUDF]]] = {
        "mlpipeline": PayloadTransformer,
        "preprocess": PreprocessUDF,
        "inference": InferenceUDF,
        "postprocess": PostprocessUDF,
        "druidtrainer": DruidTrainerUDF,
        "promtrainer": PromTrainerUDF,
        "staticthresh": StaticThresholdUDF,
        "rdstrainer": RDSTrainerUDF,
    }

    @classmethod
    def get_udf_cls(cls, udf_name: str) -> type[nl_udf_t]:
        """
        Get the UDF class.

        Args:
            udf_name: Name of the UDF;
                possible values: preprocess, inference, postprocess, trainer

        Returns
        -------
            UDF class

        Raises
        ------
            ValueError: If the UDF name is invalid
        """
        if udf_name == "trainer":
            warnings.warn(
                "UDF name 'trainer' is deprecated. Use 'druidtrainer' or 'promtrainer' instead."
            )
            udf_name = "druidtrainer"

        try:
            return cls._UDF_MAP[udf_name]
        except KeyError as err:
            _msg = f"Invalid UDF name: {udf_name}"
            _LOGGER.critical(_msg)
            raise ValueError(_msg) from err

    @classmethod
    def get_udf_instance(cls, udf_name: str, **kwargs) -> nl_udf_t:
        """
        Get the UDF instance.

        Args:
            udf_name: Name of the UDF;
                possible values: preprocess, inference, postprocess, trainer

        Returns
        -------
            UDF instance

        Raises
        ------
            ValueError: If the UDF name is invalid
        """
        udf_cls = cls.get_udf_cls(udf_name)
        return udf_cls(**kwargs)


class ServerFactory:
    """Factory class to fetch the right pynumaflow function server/mapper."""

    from pynumaflow.mapper import MapServer, MapMultiprocServer, MapAsyncServer

    _SERVER_MAP: ClassVar[dict] = {
        "sync": MapServer,
        "async": MapAsyncServer,
        "multiproc": MapMultiprocServer,
    }

    @classmethod
    def get_server_cls(cls, server_name: str):
        """
        Get the server class.

        Args:
            server_name: Name of the server;
                possible values: sync, async, multiproc

        Returns
        -------
            Server class

        Raises
        ------
            ValueError: If the server name is invalid
        """
        try:
            return cls._SERVER_MAP[server_name]
        except KeyError as err:
            _msg = f"Invalid server name: {server_name}"
            _LOGGER.critical(_msg)
            raise ValueError(_msg) from err

    @classmethod
    def get_server_instance(cls, server_name: str, **kwargs):
        """
        Get the server/mapper instance.

        Args:
            server_name: Name of the server;
                possible values: sync, async, multiproc

        Returns
        -------
            Server instance

        Raises
        ------
            ValueError: If the server name is invalid
        """
        server_cls = cls.get_server_cls(server_name)
        return server_cls(**kwargs)
