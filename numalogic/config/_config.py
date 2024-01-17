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
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class ModelInfo:
    """Schema for defining the model/estimator.

    Args:
    ----
        name: name of the model; this should map to a supported list of models
              mentioned in the factory file
        conf: kwargs for instantiating the model class
        stateful: flag indicating if the model is stateful or not
    """

    name: str = MISSING
    conf: dict[str, Any] = field(default_factory=dict)
    stateful: bool = True


# TODO add this in the right config
@dataclass
class JitterConf:
    """
    Schema for defining the jitter config to solve the Thundering Herd problem.

    Args:
    ----
        jitter_sec: Jitter in seconds
        jitter_steps_sec: Step interval value (in secs) for jitter_sec value (default = 120 sec)
    """

    jitter_sec: int = 30 * 60
    jitter_steps_sec: int = 2 * 60


@dataclass
class RegistryInfo:
    """Registry config base class.

    Args:
    ----
        name: name of the registry
        conf: kwargs for instantiating the model class
    """

    name: str
    model_expiry_sec: int
    jitter_conf: JitterConf = field(default_factory=JitterConf)
    extra_param: dict[str, Any] = field(default_factory=dict)


@dataclass
class LightningTrainerConf:
    """Schema for defining the Pytorch Lightning trainer behavior.

    More details on the arguments are provided here:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
    """

    accelerator: str = "auto"
    max_epochs: int = 50
    logger: bool = True
    log_freq: int = 5
    check_val_every_n_epoch: int = 5
    enable_checkpointing: bool = False
    enable_progress_bar: bool = False
    enable_model_summary: bool = True


@dataclass
class TrainerConf:
    """Schema for defining the trainer config."""

    train_hours: int = 24 * 8  # 8 days worth of data
    min_train_size: int = 2000
    retrain_freq_hr: int = 24
    retry_sec: int = 600  # 10 min
    batch_size: int = 64
    data_freq_sec: int = 60
    # TODO: Support trainer based transform models
    max_value_map: Optional[dict[str, float]] = None
    pltrainer_conf: LightningTrainerConf = field(default_factory=LightningTrainerConf)


@dataclass
class NumalogicConf:
    """Top level config schema for numalogic."""

    model: ModelInfo = field(default_factory=ModelInfo)
    trainer: TrainerConf = field(default_factory=TrainerConf)
    preprocess: list[ModelInfo] = field(default_factory=list)
    threshold: ModelInfo = field(default_factory=lambda: ModelInfo(name="StdDevThreshold"))
    postprocess: ModelInfo = field(
        default_factory=lambda: ModelInfo(name="TanhNorm", stateful=False)
    )


@dataclass
class DataConnectorConf:
    source: str
