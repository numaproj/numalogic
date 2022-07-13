from abc import ABCMeta, abstractmethod
from typing import Sequence, Any

from numalogic.tools.types import Artifact


class ArtifactManager(metaclass=ABCMeta):
    """
    Abstract base class for artifact save, load and delete.

    :param uri: server/connection uri
    """

    def __init__(self, uri: str):
        self.uri = uri

    @abstractmethod
    def load(
        self, skeys: Sequence[str], dkeys: Sequence[str], latest: bool = True, version: str = None
    ) -> Artifact:
        """
        Loads the desired artifact from registry/model-backend and returns it.

        :param skeys: static key fields as list/tuple of strings
        :param dkeys: dynamic key fields as list/tuple of strings
        :param latest: boolean field to determine if latest version is desired or not
        :param version: explicit artifact version
        """
        pass

    @abstractmethod
    def save(
        self, skeys: Sequence[str], dkeys: Sequence[str], artifact: Artifact, **metadata
    ) -> Any:
        """
        Saves the artifact into registry/model-backend and updates version if supported.

        :param skeys: static key fields as list/tuple of strings
        :param dkeys: dynamic key fields as list/tuple of strings
        :param artifact: artifact to be saved
        :param metadata: additional metadata surrounding the artifact that needs to be saved
        """
        pass

    @abstractmethod
    def delete(self, skeys: Sequence[str], dkeys: Sequence[str], version: str) -> None:
        """
        Deletes the artifact with a specified version from registry/model-backend.

        :param skeys: static key fields as list/tuple of strings
        :param dkeys: dynamic key fields as list/tuple of strings
        :param version: explicit artifact version
        """
        pass
