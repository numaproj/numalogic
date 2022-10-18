from ml_steps.udf import inference, postprocess, preprocess, train


class HandlerFactory:
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

        raise NotImplementedError(f"Invalid step provided: {step}")
