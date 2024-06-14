from typing import ClassVar, Union

from numalogic.config import RegistryInfo, ModelInfo
from numalogic.config.factory import _ObjectFactory
from numalogic.tools.exceptions import UnknownConfigArgsError


class RegistryFactory(_ObjectFactory):
    """Factory class to create test_registry instances."""

    _CLS_SET: ClassVar[frozenset] = frozenset(
        {"RedisRegistry", "MLflowRegistry", "DynamoDBRegistry"}
    )

    def get_instance(self, object_info: Union[ModelInfo, RegistryInfo]):
        import nlregistry as reg

        try:
            _cls = getattr(reg, object_info.name)
        except AttributeError as err:
            if object_info.name in self._CLS_SET:
                raise ImportError(
                    "Please install the required dependencies for the test_registry "
                    "you want to use."
                ) from err
            raise UnknownConfigArgsError(
                f"Invalid model info instance provided: {object_info}"
            ) from err
        return _cls(**object_info.conf)

    @classmethod
    def get_cls(cls, name: str):
        import nlregistry as reg

        try:
            return getattr(reg, name)
        except AttributeError as err:
            if name in cls._CLS_SET:
                raise ImportError(
                    "Please install the required dependencies for the test_registry "
                    "you want to use."
                ) from err
            raise UnknownConfigArgsError(
                f"Invalid name provided for RegistryFactory: {name}"
            ) from None
