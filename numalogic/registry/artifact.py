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


from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Any, Dict

from numalogic.tools.types import Artifact


@dataclass
class ArtifactData:
    artifact: Artifact
    metadata: Dict[str, Any]
    extras: Dict[str, Any]


class ArtifactManager(metaclass=ABCMeta):
    """
    Abstract base class for artifact save, load and delete.

    :param uri: server/connection uri
    """

    __slots__ = ("uri",)

    def __init__(self, uri: str):
        self.uri = uri

    @abstractmethod
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
        pass

    @abstractmethod
    def save(
        self, skeys: Sequence[str], dkeys: Sequence[str], artifact: Artifact, **metadata
    ) -> Any:
        r"""
        Saves the artifact into mlflow registry and updates version.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            artifact: primary artifact to be saved
            metadata: additional metadata surrounding the artifact that needs to be saved
        """
        pass

    @abstractmethod
    def delete(self, skeys: Sequence[str], dkeys: Sequence[str], version: str) -> None:
        """
        Deletes the artifact with a specified version from mlflow registry.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: explicit artifact version
        """
        pass
