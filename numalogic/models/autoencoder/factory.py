from numalogic.models.autoencoder.pipeline import AutoencoderPipeline, SparseAEPipeline


class ModelPlFactory:
    _pipelines = {"ae": AutoencoderPipeline, "ae_sparse": SparseAEPipeline}

    @classmethod
    def get_pl_cls(cls, plname: str):
        pl_cls = cls._pipelines.get(plname)

        if not pl_cls:
            raise NotImplementedError(f"Unsupported pl name provided: {plname}")

        return pl_cls

    @classmethod
    def get_pl_obj(cls, plname: str, *args, **kwargs):
        pl_cls = cls.get_pl_cls(plname)
        return pl_cls(*args, **kwargs)
