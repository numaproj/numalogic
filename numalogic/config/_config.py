from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from omegaconf import MISSING


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
