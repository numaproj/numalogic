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


from dataclasses import dataclass
from typing import Sequence, Any, Generic, TypeVar, Union

from numalogic.tools.types import artifact_t, S_KEYS, D_KEYS

META_T = TypeVar("META_T", bound=dict[str, Union[str, list, dict]])
EXTRA_T = TypeVar("EXTRA_T", bound=dict[str, Union[str, list, dict]])


@dataclass
class ArtifactData:
    __slots__ = ("artifact", "metadata", "extras")

    artifact: artifact_t
    metadata: META_T
    extras: EXTRA_T


A_D = TypeVar("A_D", bound=ArtifactData, covariant=True)
M_K = TypeVar("M_K", bound=str)


class ArtifactManager(Generic[S_KEYS, D_KEYS, A_D]):
    """
    Abstract base class for artifact save, load and delete.

    :param uri: server/connection uri
    """

    __slots__ = ("uri",)

    def __init__(self, uri: str):
        self.uri = uri

    def load(
        self, skeys: Sequence[str], dkeys: Sequence[str], latest: bool = True, version: str = None
    ) -> ArtifactData:
        """
        Loads the desired artifact from mlflow registry and returns it.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            latest: boolean field to determine if latest version is desired or not
            version: explicit artifact version
        """
        raise NotImplementedError("Please implement this method!")

    def save(
        self, skeys: Sequence[str], dkeys: Sequence[str], artifact: artifact_t, **metadata
    ) -> Any:
        r"""
        Saves the artifact into mlflow registry and updates version.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            artifact: primary artifact to be saved
            metadata: additional metadata surrounding the artifact that needs to be saved
        """
        raise NotImplementedError("Please implement this method!")

    def delete(self, skeys: Sequence[str], dkeys: Sequence[str], version: str) -> None:
        """
        Deletes the artifact with a specified version from mlflow registry.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: explicit artifact version
        """
        raise NotImplementedError("Please implement this method!")

    @staticmethod
    def construct_key(skeys: Sequence[str], dkeys: Sequence[str]) -> str:
        """
        Returns a single key comprising static and dynamic key fields.
        Override this method if customization is needed.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings

        Returns:
            key
        """
        _static_key = ":".join(skeys)
        _dynamic_key = ":".join(dkeys)
        return "::".join([_static_key, _dynamic_key])


class ArtifactCache(Generic[M_K, A_D]):
    r"""
    Base class for all artifact caches.
    Caches support saving, loading and deletion, but not artifact versioning.

    Args:
        cachesize: size of the cache
        ttl: time to live for each item in the cache
    """
    __slots__ = ("_cachesize", "_ttl")

    def __init__(self, cachesize: int, ttl: int):
        self._cachesize = cachesize
        self._ttl = ttl

    @property
    def cachesize(self):
        return self._cachesize

    @property
    def ttl(self):
        return self._ttl

    def load(self, key: str) -> ArtifactData:
        r"""
        Returns the stored ArtifactData object from the cache.
        Implement this method for your custom cache.
        """
        raise NotImplementedError("Please implement this method!")

    def save(self, key: str, artifact: ArtifactData) -> None:
        r"""
        Saves the ArtifactData object into the cache.
        Implement this method for your custom cache.
        """
        raise NotImplementedError("Please implement this method!")

    def delete(self, key: str) -> None:
        r"""
        Deletes the ArtifactData object from the cache.
        Implement this method for your custom cache.
        """
        raise NotImplementedError("Please implement this method!")
