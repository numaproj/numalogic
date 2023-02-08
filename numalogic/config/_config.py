# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from omegaconf import MISSING


@dataclass
class ModelInfo:
    """
    Schema for defining the model/estimator.

    Args:
        name: name of the model; this should map to a supported list of models
              mentioned in the factory file
        conf: kwargs for instantiating the model class
        stateful: flag indicating if the model is stateful or not
    """

    name: str = MISSING
    conf: Dict[str, Any] = field(default_factory=dict)
    stateful: bool = True


@dataclass
class RegistryConf:
    # TODO implement this
    """
    Registry config base class
    """
    pass


@dataclass
class LightningTrainerConf:
    """
    Schema for defining the Pytorch Lightning trainer behavior.

    More details on the arguments are provided here:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
    """

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
    """
    Top level config schema for numalogic.
    """

    model: ModelInfo = field(default_factory=ModelInfo)
    trainer: LightningTrainerConf = field(default_factory=LightningTrainerConf)
    registry: RegistryConf = field(default_factory=RegistryConf)
    preprocess: List[ModelInfo] = field(default_factory=list)
    threshold: ModelInfo = field(default_factory=ModelInfo)
    postprocess: ModelInfo = field(default_factory=ModelInfo)
