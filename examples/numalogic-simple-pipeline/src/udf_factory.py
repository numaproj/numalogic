from src.udf import inference, postprocess, preprocess, train, threshold


class HandlerFactory:
    """Factory class to return the handler for the given step."""

    @classmethod
    def get_handler(cls, step: str):
        if step == "preprocess":
            return preprocess

        if step == "inference":
            return inference

        if step == "postprocess":
            return postprocess

        if step == "train":
            return train

        if step == "threshold":
            return threshold

        raise ValueError(f"Invalid step provided: {step}")
