from typing import ClassVar

from numalogic.udfs.inference import InferenceUDF
from numalogic.udfs.trainer import TrainerUDF
from numalogic.udfs.preprocess import PreprocessUDF
from numalogic.udfs.postprocess import PostprocessUDF


class UDFFactory:
    _UDF_MAP: ClassVar[dict] = {
        "preprocess": PreprocessUDF,
        "inference": InferenceUDF,
        "trainer": TrainerUDF,
        "postprocess": PostprocessUDF,
    }

    @classmethod
    def get_udf_cls(cls, udf_name: str):
        return cls._UDF_MAP[udf_name]
