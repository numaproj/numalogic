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

from typing import Optional

from cachetools import TTLCache

from numalogic.registry.artifact import ArtifactCache, ArtifactData
from numalogic.tools.types import Singleton


class LocalLRUCache(ArtifactCache, metaclass=Singleton):
    r"""
    A local in-memory LRU cache with per item Time-to-live support.

    Args:
        cachesize: Size of the cache,
                   i.e. number of elements the cache can hold
        ttl: Time to live for each item in seconds
    """
    __cache: Optional[TTLCache] = None

    def __init__(self, cachesize: int = 512, ttl: int = 300):
        super().__init__(cachesize, ttl)
        if not self.__cache:
            self.__cache = TTLCache(maxsize=cachesize, ttl=ttl)

    def load(self, artifact_key: str) -> ArtifactData:
        return self.__cache.get(artifact_key)

    def save(self, key: str, artifact: ArtifactData) -> None:
        self.__cache[key] = artifact

    def delete(self, key: str) -> Optional[ArtifactData]:
        return self.__cache.pop(key, default=None)
