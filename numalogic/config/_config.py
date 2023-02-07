from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from omegaconf import OmegaConf, MISSING


@dataclass
class ModelInfo:
    name: str = MISSING
    conf: Dict[str, Any] = field(default_factory=dict)
    stateful: bool = True


@dataclass
class RegistryConf:
    pass


@dataclass
class LightningTrainerConf:
    max_epochs: int = 100
    logger: bool = False
    check_val_every_n_epoch: int = 5
    log_every_n_steps: int = 20
    enable_checkpointing: bool = False
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    limit_val_batches: bool = 0
    callbacks: Optional[Any] = None


@dataclass
class NumalogicConf:
    model: ModelInfo = field(default_factory=ModelInfo)
    trainer: LightningTrainerConf = field(default_factory=LightningTrainerConf)
    registry: RegistryConf = field(default_factory=RegistryConf)
    preprocess: List[ModelInfo] = field(default_factory=list)
    threshold: ModelInfo = field(default_factory=ModelInfo)
    postprocess: ModelInfo = field(default_factory=ModelInfo)


def my_app():
    from numalogic.config.factory import (
        ModelFactory,
        PreprocessFactory,
        PostprocessFactory,
        ThresholdFactory,
    )
    from numalogic.models.autoencoder import AutoencoderTrainer
    import os

    os.environ["OC_CAUSE"] = "1"

    schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
    print(type(schema))

    conf = OmegaConf.load("../../conf/cfg.yaml")
    merged = OmegaConf.merge(schema, conf)
    print(merged)

    model_info: ModelInfo = OmegaConf.to_object(merged["model"])

    factory = ModelFactory()
    model = factory.get_model_instance(model_info)
    print(model)

    trainer_config = merged.trainer
    trainer = AutoencoderTrainer(**trainer_config)
    print(trainer)

    postproc_factory = PostprocessFactory()
    postproc_clf = postproc_factory.get_model_instance(merged.postprocess)
    print(postproc_clf)

    threshold_factory = ThresholdFactory()
    thresh_clf = threshold_factory.get_model_instance(merged.threshold)
    print(thresh_clf)

    preproc_factory = PreprocessFactory()
    for _cfg in merged.preprocess:
        print(preproc_factory.get_model_instance(_cfg))
    # preproc_factory.get_model_instance(merged.preprocess)


if __name__ == "__main__":
    my_app()
