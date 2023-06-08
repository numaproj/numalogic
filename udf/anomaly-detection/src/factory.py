from src.udf import Preprocess, Inference, Threshold, Postprocess
from src.udsink import Train


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

        if step == "train":
            return Train().run

        raise NotImplementedError(f"Invalid step provided: {step}")
