from src.udf import Preprocess, Inference, Postprocess, Trainer, Threshold
from numalogic.numaflow import NumalogicUDF


class UDFFactory:
    """Factory class to return the handler for the given step."""

    _UDF_MAP = {
        "preprocess": Preprocess,
        "inference": Inference,
        "postprocess": Postprocess,
        "train": Trainer,
        "threshold": Threshold,
    }

    @classmethod
    def get_handler(cls, step: str) -> NumalogicUDF:
        """Return the handler for the given step."""
        udf_cls = cls._UDF_MAP[step]
        return udf_cls()
