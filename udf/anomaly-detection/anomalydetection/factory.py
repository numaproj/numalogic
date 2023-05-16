from typing import Callable, Union

from pynumaflow.function import Messages
from pynumaflow.sink import Responses

from anomalydetection.udf import window, aggregate, preprocess
from anomalydetection.udf.keying import keying


class HandlerFactory:
    @classmethod
    def get_handler(cls, step: str):
        if step == "keying":
            return keying

        if step == "aggregate":
            return aggregate

        if step == "window":
            return window

        if step == "preprocess":
            return preprocess

        raise NotImplementedError(f"Invalid step provided: {step}")
