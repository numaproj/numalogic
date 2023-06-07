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


from typing import Union, TypeVar
from collections.abc import Sequence
from sklearn.base import BaseEstimator
from torch import nn


try:
    from redis.client import AbstractRedis
except ImportError:
    pass
else:
    redis_client_t = TypeVar("redis_client_t", bound=AbstractRedis, covariant=True)

artifact_t = TypeVar("artifact_t", bound=Union[nn.Module, BaseEstimator], covariant=True)
META_T = TypeVar("META_T", bound=dict[str, Union[str, float, int, list, dict]])
META_VT = TypeVar("META_VT", str, int, float, list, dict)
EXTRA_T = TypeVar("EXTRA_T", bound=dict[str, Union[str, list, dict]])
KEYS = TypeVar("KEYS", bound=Sequence[str], covariant=True)


class Singleton(type):
    r"""Helper metaclass to use as a Singleton class."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
