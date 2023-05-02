import json
import logging
import socket
import time
from typing import Optional, Sequence, Dict, Any

import redis
from redis.backoff import ExponentialBackoff
from redis.client import AbstractRedis
from redis.exceptions import RedisClusterException, RedisError
from redis.retry import Retry

from numalogic.registry import ArtifactManager, ArtifactData
from numalogic.registry._serialize import loads, dumps
from numalogic.tools.exceptions import ModelKeyNotFound
from numalogic.tools.types import Artifact

_LOGGER = logging.getLogger()


def get_ipv4_by_hostname(hostname: str, port=0) -> list:
    return list(
        idx[4][0]
        for idx in socket.getaddrinfo(hostname, port)
        if idx[0] is socket.AddressFamily.AF_INET and idx[1] is socket.SocketKind.SOCK_RAW
    )


def is_host_reachable(hostname: str, port=None, max_retries=5, sleep_sec=5) -> bool:
    retries = 0
    assert max_retries >= 1, "Max retries has to be at least 1"

    while retries < max_retries:
        try:
            get_ipv4_by_hostname(hostname, port)
        except socket.gaierror as ex:
            retries += 1
            _LOGGER.warning(
                "Failed to resolve hostname: %s: error: %r", hostname, ex, exc_info=True
            )
            time.sleep(sleep_sec)
        else:
            return True
    _LOGGER.error("Failed to resolve hostname: %s even after retries!")
    return False


class RedisRegistry(ArtifactManager):
    """
    Model saving and loading using Redis Registry. The RedisRegistry takes in any Redis client or
    can generates a new client using host, port and password arguemnts.
    Args:
        client: Take in the reids client already established/created
        host: hostname for redis client
        port: port for redis client
        password: password for redis client
        decode_responses: decode the response that we get from redis (default = False)
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
    >>> registry.load(skeys, dkeys, artifact=model)
    """

    __slots__ = ("client", "ttl", "pipe")
    _HOST = None

    def __new__(
        cls,
        client: redis.client = None,
        host: str = None,
        port: str = None,
        password: str = None,
        decode_responses: bool = False,
        ttl: int = 50000,
        *args,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        if (not cls._HOST) or (cls._HOST != host):
            cls._HOST = host
        return instance

    def __init__(
        self,
        client: Optional[AbstractRedis] = None,
        host: str = None,
        port: str = None,
        password: str = None,
        decode_responses: bool = False,
        ttl: int = 50000,
    ):
        super().__init__(host)
        if (client and (host and port)) or (not client and not (host and port)):
            raise ValueError("Either One of 'client' or 'host and port' information is missing")
        if client:
            self.client = client
        else:
            redis_params = {
                "host": host,
                "port": port,
                "password": password,
                "decode_responses": decode_responses,
            }
            _LOGGER.info("Redis params: %s", json.dumps(redis_params, indent=4))

            if not is_host_reachable(host, port):
                _LOGGER.error("Redis Cluster is unreachable after retries!")

            retry = Retry(
                ExponentialBackoff(cap=2, base=1),
                3,
                supported_errors=(ConnectionError, TimeoutError, RedisClusterException, RedisError),
            )
            pool = redis.ConnectionPool(**redis_params, retry=retry)
            self.client = redis.Redis(connection_pool=pool)
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

    def __load_metadata(self, model_key: str) -> Dict[str, Any]:
        metadata = None
        if self.client.hexists(name=model_key, key="metadata"):
            serialized_metadata = self.client.hget(name=model_key, key="metadata")
            metadata = loads(serialized_metadata)
        return metadata

    def __save_metadata(self, metadata: Dict[str, Any], key: str):
        serialized_metadata = dumps(metadata)
        self.client.hset(
            name=key,
            mapping={"metadata": serialized_metadata},
        )

    def __get_model_key(
        self, production: bool, version: str, skeys: Sequence[str], dkeys: Sequence[str]
    ) -> str:
        if production:
            production_key = self.construct_key(skeys, dkeys, ["PRODUCTION"])
            if not self.client.exists(production_key):
                raise ModelKeyNotFound(
                    "Production key: %s, Not Found !!!\n Exiting.....",
                    production_key,
                )
            model_key = self.client.get(production_key)
            if not self.client.exists(model_key):
                raise ModelKeyNotFound(
                    "Production is pointing to: %s.\n Could not find model key with key: %d",
                    production_key,
                    model_key,
                )
        else:
            model_key = self.construct_key(skeys, dkeys, [version])
            if not self.client.exists(model_key):
                raise ModelKeyNotFound("Could not find model key with key: %d", model_key)
        return model_key

    def __save_artifact(
        self, artifact: Artifact, skeys: Sequence[str], dkeys: Sequence[str], version: str
    ):
        new_version_key = self.construct_key(skeys, dkeys, [version])
        production_key = self.construct_key(skeys, dkeys, ["PRODUCTION"])
        self.client.set(name=production_key, value=new_version_key)
        _LOGGER.info(
            "Setting Production key : %d ,to this new key = %s", production_key, new_version_key
        )
        serialized_object = dumps(deserialized_object=artifact)
        self.client.hset(
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
        production: bool = True,
        version: str = None,
    ) -> Optional[ArtifactData]:
        """
        Saves the artifact into mlflow registry and updates version.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            production: load the model in production stage
            version: version to load

        Returns:
            mlflow ModelVersion instance
        """
        try:
            if (production and version) or (not production and not version):
                raise ValueError("Either One of 'latest' or 'version' needed in load method call")
            model_key = self.__get_model_key(production, version, skeys, dkeys)
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
        except Exception as ex:
            _LOGGER.exception("Unexpected error: %s", ex)
            return None

    def save(
        self,
        skeys: Sequence[str],
        dkeys: Sequence[str],
        artifact: Artifact,
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
            new_version_key = self.__save_artifact(artifact, skeys, dkeys, str(version))
            if metadata:
                self.__save_metadata(metadata, new_version_key)
            self.client.expire(name=new_version_key, time=self.ttl)
            _LOGGER.info("Model is successfully with the key = %s", new_version_key)
            return str(version)
        except Exception as ex:
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
            del_key = self.construct_key(skeys, dkeys, [version])
            if self.client.exists(del_key):
                self.client.delete(del_key)
                _LOGGER.info("Model with the key = %s, deleted successfully", del_key)
            else:
                raise ModelKeyNotFound(
                    "Key to delete: %s, Not Found !!!\n Exiting.....",
                    del_key,
                )
        except ModelKeyNotFound as model_key_error:
            _LOGGER.exception("Missing Key: %s", model_key_error)
            return None
        except Exception as ex:
            _LOGGER.exception("Unexpected error: %s", ex)
            return None
