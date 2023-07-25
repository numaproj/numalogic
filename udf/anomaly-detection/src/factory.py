from src.udf import Preprocess, Inference, Threshold, Postprocess
from src.udf.trainer import Trainer


class HandlerFactory:
    @classmethod
    def get_handler(cls, step: str):
        if step == "preprocess":
            return Preprocess().run

        if step == "inference":
            return Inference().run

        if step == "threshold":
            return Threshold().run

        if step == "postprocess":
            return Postprocess().run

        if step == "trainer":
            return Trainer().run

        raise NotImplementedError(f"Invalid step provided: {step}")
