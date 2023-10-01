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
import random
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Union, Optional

from numalogic.tools.exceptions import ConfigError
from numalogic.tools.types import artifact_t, KEYS, META_T, META_VT, EXTRA_T, state_dict_t


@dataclass
class ArtifactData:
    """
    Dataclass to hold the artifact, its metadata and other extra info.

    Args:
    ----
        artifact: artifact to be saved; can be a model instance, a state_dict.
        metadata: additional metadata surrounding the artifact that needs to be saved.
        extras: any other extra information that needs to be saved.

    """

    __slots__ = ("artifact", "metadata", "extras")

    artifact: Union[artifact_t, state_dict_t]
    metadata: META_T
    extras: EXTRA_T


A_D = TypeVar("A_D", bound=ArtifactData, covariant=True)
M_K = TypeVar("M_K", bound=str)


class ArtifactManager(Generic[KEYS, A_D]):
    """Abstract base class for artifact save, load and delete.

    Args:
    ----
        uri: server/connection uri
    """

    _STORETYPE = "registry"

    __slots__ = ("uri",)

    def __init__(self, uri: str):
        self.uri = uri

    def load(
        self, skeys: KEYS, dkeys: KEYS, latest: bool = True, version: Optional[str] = None
    ) -> ArtifactData:
        """Loads the desired artifact from mlflow registry and returns it.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            latest: boolean field to determine if latest version is desired or not
            version: explicit artifact version.
        """
        raise NotImplementedError("Please implement this method!")

    def save(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        artifact: Union[artifact_t, state_dict_t],
        **metadata: META_VT
    ) -> Any:
        r"""Saves the artifact into mlflow registry and updates version.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            artifact: primary artifact to be saved
            metadata: additional metadata surrounding the artifact that needs to be saved.
        """
        raise NotImplementedError("Please implement this method!")

    def delete(self, skeys: KEYS, dkeys: KEYS, version: str) -> None:
        """Deletes the artifact with a specified version from mlflow registry.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: explicit artifact version.
        """
        raise NotImplementedError("Please implement this method!")

    @staticmethod
    def is_artifact_stale(artifact_data: ArtifactData, freq_hr: int) -> bool:
        """Returns whether the given artifact is stale or not, i.e. if
        more time has elapsed since it was last retrained.

        Args:
        ----
            artifact_data: ArtifactData object to look into
            freq_hr: Frequency of retraining in hours.
        """
        raise NotImplementedError("Please implement this method!")

    @staticmethod
    def construct_key(skeys: KEYS, dkeys: KEYS) -> str:
        """Returns a single key comprising static and dynamic key fields.
        Override this method if customization is needed.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings.

        Returns
        -------
            key
        """
        _static_key = ":".join(skeys)
        _dynamic_key = ":".join(dkeys)
        return "::".join([_static_key, _dynamic_key])


def _apply_jitter(ts: int, jitter_sec: int, jitter_steps_sec: int):
    """
        Applies jitter to the ttl value to solve Thundering Herd problem.
    z
        Note: Jitter izs not applied if jitter_sec and jitter_steps_sec are both 0.
    """
    if jitter_sec == jitter_steps_sec == 0:
        return ts
    if jitter_sec < jitter_steps_sec:
        raise ConfigError("jitter_sec should be at least 60*jitter_steps_sec")
    begin = ts if ts - jitter_sec < 0 else ts - jitter_sec
    end = ts + jitter_sec + 1
    return random.randrange(begin, end, jitter_steps_sec)


class ArtifactCache(Generic[M_K, A_D]):
    r"""Base class for all artifact caches.
    Caches support saving, loading and deletion, but not artifact versioning.

    Args:
    ----
        cachesize: size of the cache
        ttl: time to live for each item in the cache
        jitter_sec: jitter in seconds to add to the ttl (to solve Thundering Herd problem)
        jitter_steps_sec: Step interval value (in mins) for jitter_sec value
    """

    _STORETYPE = "cache"

    __slots__ = ("_cachesize", "_ttl", "jitter_sec", "jitter_steps_sec")

    def __init__(self, cachesize: int, ttl: int, jitter_sec: int, jitter_steps_sec: int):
        self._cachesize = cachesize
        self._ttl = _apply_jitter(ts=ttl, jitter_sec=jitter_sec, jitter_steps_sec=jitter_steps_sec)

    @property
    def cachesize(self):
        return self._cachesize

    @property
    def ttl(self):
        return self._ttl

    def load(self, key: str) -> ArtifactData:
        r"""Returns the stored ArtifactData object from the cache.
        Implement this method for your custom cache.
        """
        raise NotImplementedError("Please implement this method!")

    def save(self, key: str, artifact: ArtifactData) -> None:
        r"""Saves the ArtifactData object into the cache.
        Implement this method for your custom cache.
        """
        raise NotImplementedError("Please implement this method!")

    def delete(self, key: str) -> None:
        r"""Deletes the ArtifactData object from the cache.
        Implement this method for your custom cache.
        """
        raise NotImplementedError("Please implement this method!")

    def clear(self) -> None:
        r"""Clears the cache.
        Implement this method for your custom cache.
        """
        raise NotImplementedError("Please implement this method!")
