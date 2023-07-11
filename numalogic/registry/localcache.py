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

from copy import deepcopy
from threading import Lock
from typing import Optional

from cachetools import TTLCache

from numalogic.registry.artifact import ArtifactCache, ArtifactData
from numalogic.tools.types import Singleton


class LocalLRUCache(ArtifactCache, metaclass=Singleton):
    r"""A local in-memory LRU cache registry with per artifact Time-to-live support.

    Args:
    ----
        cachesize: Size of the cache,
                   i.e. number of elements the cache can hold
        ttl: Time to live for each item in seconds
    """

    __cache: Optional[TTLCache] = None

    def __init__(self, cachesize: int = 512, ttl: int = 300):
        super().__init__(cachesize, ttl)
        if not self.__cache:
            self.__cache = TTLCache(maxsize=cachesize, ttl=ttl)
        self.__lock = Lock()

    def __contains__(self, artifact_key: str) -> bool:
        """Check if an artifact is in the cache."""
        return artifact_key in self.__cache

    def load(self, artifact_key: str) -> Optional[ArtifactData]:
        """
        Load an artifact from the cache.

        Args:
        ----
            artifact_key: The key of the artifact to load.

        Returns
        -------
            The artifact data instance if found, None otherwise.
        """
        with self.__lock:
            return self.__cache.get(artifact_key)

    def save(self, key: str, artifact: ArtifactData) -> None:
        """
        Save an artifact to the cache.

        Args:
        ----
            key: The key of the artifact to save.
            artifact: The artifact data instance to save.
        """
        artifact = deepcopy(artifact)
        artifact.extras["source"] = self._STORETYPE
        with self.__lock:
            self.__cache[key] = artifact

    def delete(self, key: str) -> Optional[ArtifactData]:
        """
        Delete an artifact from the cache.

        Args:
        ----
            key: The key of the artifact to delete.

        Returns
        -------
            The deleted artifact data instance if found, None otherwise.
        """
        with self.__lock:
            return self.__cache.pop(key, default=None)

    def clear(self) -> None:
        """Clears the whole cache."""
        with self.__lock:
            self.__cache.clear()

    def keys(self) -> list[str]:
        """Returns the current keys of the cache."""
        return list(_key for _key in self.__cache)
