from src.udf import Preprocess, Inference, Postprocess, Trainer, Threshold


class UDFFactory:
    """Factory class to return the handler for the given step."""

    _UDF_MAP = {
        "preprocess": Preprocess(),
        "inference": Inference(),
        "postprocess": Postprocess(),
        "train": Trainer(),
        "threshold": Threshold(),
    }

    @classmethod
    def get_handler(cls, step: str):
        """Return the handler for the given step."""
        return cls._UDF_MAP[step]
