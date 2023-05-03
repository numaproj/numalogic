import logging
import time
from typing import Optional, Any
from collections.abc import Sequence

from redis.client import Redis, Pipeline
from redis.exceptions import RedisError

from numalogic.registry import ArtifactManager, ArtifactData
from numalogic.registry.serialize import loads, dumps
from numalogic.tools.exceptions import ModelKeyNotFound
from numalogic.tools.types import artifact_t

_LOGGER = logging.getLogger()


class RedisRegistry(ArtifactManager):
    """
    Model saving and loading using Redis Registry.
    Args:
        client: Take in the reids client already established/created
        ttl: Total Time to Live for the key when saving in redis (dafault = 5000)

    Examples
    --------
    >>> import redis
    >>> from numalogic.models.autoencoder.variants import VanillaAE
    >>> from numalogic.registry.redis_registry import RedisRegistry
    >>> r = redis.StrictRedis(host='127.0.0.1', port=6379)
    >>> cli = r.client()
    >>> registry = RedisRegistry(client=cli)
    >>> skeys = ['c', 'a']
    >>> dkeys = ['d', 'a']
    >>> model = VanillaAE(10)
    >>> registry.save(skeys, dkeys, artifact=model, **{'lr': 0.01})
    >>> registry.load(skeys, dkeys)
    """

    __slots__ = ("client", "ttl")
    _HOST = None

    def __new__(
        cls,
        client: Redis = None,
        ttl: int = 604800,
        *args,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        if not cls._HOST:
            cls._HOST = "None"
        return instance

    def __init__(
        self,
        client: Redis = None,
        ttl: int = 604800,
    ):
        super().__init__("None")
        self.client = client
        if not client:
            raise ValueError("Missing Redis Client")
        self.ttl = ttl

    @staticmethod
    def construct_key(*keys: [Sequence[str]]) -> str:
        """
        Returns a single key comprising static and dynamic key fields.
        Args:
            keys: Sequence of strings sequence combined to form key
        Returns:
            key
        """
        key = "::".join([":".join(key) for key in keys])
        return key

    @staticmethod
    def get_version(key: str) -> str:
        """
        get version number from the string
        Args:
            key: key

        Returns:
            version
        """
        version_number = key.split("::")
        return version_number[-1]

    def __load_metadata(self, model_key: str) -> dict[str, Any]:
        metadata = None
        if self.client.hexists(name=model_key, key="metadata"):
            serialized_metadata = self.client.hget(name=model_key, key="metadata")
            metadata = loads(serialized_metadata)
        return metadata

    def __save_metadata(self, pipe: Pipeline, metadata: dict[str, Any], key: str):
        serialized_metadata = dumps(metadata)
        pipe.hset(
            name=key,
            mapping={"metadata": serialized_metadata},
        )

    def __get_model_key(
        self, latest: bool, version: str, skeys: Sequence[str], dkeys: Sequence[str]
    ) -> str:
        if latest:
            production_key = self.construct_key(skeys, dkeys, ["PRODUCTION"])
            if not production_key:
                raise ModelKeyNotFound(
                    f"Production key: {production_key}, Not Found !!!\n Exiting....."
                )
            model_key = self.client.get(production_key)
            if not model_key:
                raise ModelKeyNotFound(
                    "Production key = {} is pointing to the key: {} that "
                    "is missing the redis registry".format(production_key, model_key)
                )
        else:
            model_key = self.construct_key(skeys, dkeys, [version])
            if not self.client.exists(model_key):
                raise ModelKeyNotFound("Could not find model key with key: %s" % model_key)
        return model_key

    def __save_artifact(
        self, pipe, artifact: artifact_t, skeys: Sequence[str], dkeys: Sequence[str], version: str
    ):
        new_version_key = self.construct_key(skeys, dkeys, [version])
        production_key = self.construct_key(skeys, dkeys, ["PRODUCTION"])
        pipe.set(name=production_key, value=new_version_key)
        _LOGGER.info(
            "Setting Production key : %d ,to this new key = %s", production_key, new_version_key
        )
        serialized_object = dumps(deserialized_object=artifact)
        pipe.hset(
            name=new_version_key,
            mapping={
                "model": serialized_object,
                "version": str(version),
                "timestamp": int(time.time()),
            },
        )
        return new_version_key

    def load(
        self,
        skeys: Sequence[str],
        dkeys: Sequence[str],
        latest: bool = True,
        version: str = None,
    ) -> Optional[ArtifactData]:
        """
        Saves the artifact into mlflow registry and updates version.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            latest: load the model in production stage
            version: version to load

        Returns:
            mlflow ModelVersion instance
        """
        try:
            if (latest and version) or (not latest and not version):
                raise ValueError("Either One of 'latest' or 'version' needed in load method call")
            model_key = self.__get_model_key(latest, version, skeys, dkeys)
            serialized_model, model_version, model_timestamp = self.client.hmget(
                name=model_key, keys=["model", "version", "timestamp"]
            )
            deserialized_model = loads(serialized_model)
            metadata = self.__load_metadata(model_key)
            return ArtifactData(
                artifact=deserialized_model,
                metadata=metadata,
                extras={
                    "model_timestamp": int(model_timestamp.decode()),
                    "model_version": model_version.decode(),
                },
            )
        except ModelKeyNotFound as model_key_error:
            _LOGGER.exception("Missing Key: %s", model_key_error)
            return None
        except RedisError as ex:
            _LOGGER.exception("Unexpected error: %s", ex)
            return None

    def save(
        self,
        skeys: Sequence[str],
        dkeys: Sequence[str],
        artifact: artifact_t,
        **metadata: str,
    ) -> Optional[str]:
        """
        Saves the artifact into redis registry and updates version.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            artifact: primary artifact to be saved
            metadata: additional metadata surrounding the artifact that needs to be saved

        Returns:
            model version
        """
        try:
            production_key = self.construct_key(skeys, dkeys, ["PRODUCTION"])
            version = 0
            if self.client.exists(production_key):
                _LOGGER.info("Production key exists for the model")
                version_key = self.client.get(name=production_key)
                version = int(self.get_version(version_key.decode())) + 1
            with self.client.pipeline() as pipe:
                new_version_key = self.__save_artifact(pipe, artifact, skeys, dkeys, str(version))
                if metadata:
                    self.__save_metadata(pipe, metadata, new_version_key)
                pipe.expire(name=new_version_key, time=self.ttl)
                _LOGGER.info("Model is successfully with the key = %s", new_version_key)
                pipe.execute()
                return str(version)
        except RedisError as ex:
            _LOGGER.exception("Unexpected error: %s", ex)
            return None

    def delete(self, skeys: Sequence[str], dkeys: Sequence[str], version: str) -> None:
        """
        Saves the artifact into mlflow registry and updates version.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: model version to delete
        """
        try:
            with self.client.pipeline() as pipe:
                del_key = self.construct_key(skeys, dkeys, [version])
                if pipe.exists(del_key):
                    pipe.delete(del_key)
                    _LOGGER.info("Model with the key = %s, deleted successfully", del_key)
                else:
                    raise ModelKeyNotFound(
                        "Key to delete: %s, Not Found !!!\n Exiting....." % del_key,
                    )
        except ModelKeyNotFound as model_key_error:
            _LOGGER.exception("Missing Key: %s", model_key_error)
            return None
        except RedisError as ex:
            _LOGGER.exception("Unexpected error: %s", ex)
            return None
