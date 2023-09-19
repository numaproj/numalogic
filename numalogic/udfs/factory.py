import logging
from typing import ClassVar

from pynumaflow.mapper import Mapper, MultiProcMapper, AsyncMapper

from numalogic.udfs import NumalogicUDF
from numalogic.udfs.inference import InferenceUDF
from numalogic.udfs.postprocess import PostprocessUDF
from numalogic.udfs.preprocess import PreprocessUDF
from numalogic.udfs.trainer import TrainerUDF

_LOGGER = logging.getLogger(__name__)


class UDFFactory:
    """Factory class to fetch the right UDF."""

    _UDF_MAP: ClassVar[dict[str, type[NumalogicUDF]]] = {
        "preprocess": PreprocessUDF,
        "inference": InferenceUDF,
        "postprocess": PostprocessUDF,
        "trainer": TrainerUDF,
    }

    @classmethod
    def get_udf_cls(cls, udf_name: str) -> type[NumalogicUDF]:
        try:
            return cls._UDF_MAP[udf_name]
        except KeyError as err:
            _msg = f"Invalid UDF name: {udf_name}"
            _LOGGER.critical(_msg)
            raise ValueError(_msg) from err

    @classmethod
    def get_udf_instance(cls, udf_name: str, **kwargs) -> NumalogicUDF:
        udf_cls = cls.get_udf_cls(udf_name)
        return udf_cls(**kwargs)


class ServerFactory:
    """Factory class to fetch the right pynumaflow function server/mapper."""

    _SERVER_MAP: ClassVar[dict] = {
        "sync": Mapper,
        "async": AsyncMapper,
        "multiproc": MultiProcMapper,
    }

    @classmethod
    def get_server_cls(cls, server_name: str):
        try:
            return cls._SERVER_MAP[server_name]
        except KeyError as err:
            _msg = f"Invalid server name: {server_name}"
            _LOGGER.critical(_msg)
            raise ValueError(_msg) from err

    @classmethod
    def get_server_instance(cls, server_name: str, **kwargs):
        server_cls = cls.get_server_cls(server_name)
        return server_cls(**kwargs)
