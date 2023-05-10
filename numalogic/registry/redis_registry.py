import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from redis.exceptions import RedisError

from numalogic.registry import ArtifactManager, ArtifactData, ArtifactCache
from numalogic.registry._serialize import loads, dumps
from numalogic.tools.exceptions import ModelKeyNotFound, RedisRegistryError
from numalogic.tools.types import artifact_t, redis_client_t, KEYS, META_T, META_VT

_LOGGER = logging.getLogger()


class RedisRegistry(ArtifactManager):
    """
    Model saving and loading using Redis Registry.
    Args:
        client: Take in the reids client already established/created
        ttl: Total Time to Live (in seconds) for the key when saving in redis (dafault = 604800)
        cache_registry: Cache registry to use (default = None)

    Examples
    --------
    >>> import redis
    >>> from numalogic.models.autoencoder.variants import VanillaAE
    >>> from numalogic.registry.redis_registry import RedisRegistry
    >>> ...
    >>> r = redis.StrictRedis(host='127.0.0.1', port=6379)
    >>> registry = RedisRegistry(client=r)
    >>> skeys, dkeys = ("mymetric", "ae"), ("vanilla", "seq10")
    >>> model = VanillaAE(seq_len=10)
    >>> registry.save(skeys, dkeys, artifact=model, **{'lr': 0.01})
    >>> loaded_artifact = registry.load(skeys, dkeys)
    """

    __slots__ = ("client", "ttl", "cache_registry")

    def __init__(
        self,
        client: redis_client_t,
        ttl: int = 604800,
        cache_registry: ArtifactCache = None,
    ):
        super().__init__("")
        self.client = client
        self.ttl = ttl
        self.cache_registry = cache_registry

    @staticmethod
    def construct_key(skeys: KEYS, dkeys: KEYS) -> str:
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

    @staticmethod
    def __construct_production_key(key: str):
        return RedisRegistry.construct_key(skeys=[key], dkeys=["PROD"])

    @staticmethod
    def __construct_version_key(key: str, version: str):
        return RedisRegistry.construct_key(skeys=[key], dkeys=[version])

    @staticmethod
    def get_version(key: str) -> str:
        """
        get version number from the string
        Args:
            key: full model key

        Returns:
            version
        """
        return key.split("::")[-1]

    def _load_from_cache(self, key: str) -> Optional[ArtifactData]:
        if not self.cache_registry:
            return None
        return self.cache_registry.load(key)

    def _save_in_cache(self, key: str, artifact_data: ArtifactData) -> None:
        if self.cache_registry:
            self.cache_registry.save(key, artifact_data)

    def _clear_cache(self, key: Optional[str] = None) -> Optional[ArtifactData]:
        if self.cache_registry:
            if key:
                return self.cache_registry.delete(key)
            return self.cache_registry.clear()
        return None

    def __get_artifact_data(
        self,
        model_key: str,
    ) -> ArtifactData:
        (
            serialized_artifact,
            artifact_version,
            artifact_timestamp,
            serialized_metadata,
        ) = self.client.hmget(name=model_key, keys=["artifact", "version", "timestamp", "metadata"])
        deserialized_artifact = loads(serialized_artifact)
        deserialized_metadata = None
        if serialized_metadata:
            deserialized_metadata = loads(serialized_metadata)
        return ArtifactData(
            artifact=deserialized_artifact,
            metadata=deserialized_metadata,
            extras={
                "timestamp": float(artifact_timestamp.decode()),
                "version": artifact_version.decode(),
            },
        )

    def __load_latest_artifact(self, key: str) -> ArtifactData:
        cached_artifact = self._load_from_cache(key)
        if cached_artifact:
            return cached_artifact
        production_key = self.__construct_production_key(key)
        if not self.client.exists(production_key):
            raise ModelKeyNotFound(
                f"Production key: {production_key}, Not Found !!!\n Exiting....."
            )
        model_key = self.client.get(production_key)
        _LOGGER.info("Production key, %s, is pointing to the key : %s", production_key, model_key)
        return self.__load_version_artifact(version=self.get_version(model_key.decode()), key=key)

    def __load_version_artifact(self, version: str, key: str) -> ArtifactData:
        model_key = self.__construct_version_key(key, version)
        if not self.client.exists(model_key):
            raise ModelKeyNotFound("Could not find model key with key: %s" % model_key)
        return self.__get_artifact_data(
            model_key=model_key,
        )

    def __save_artifact(
        self, pipe, artifact: artifact_t, metadata: META_T, key: KEYS, version: str
    ) -> str:
        new_version_key = self.__construct_version_key(key, version)
        production_key = self.__construct_production_key(key)
        pipe.set(name=production_key, value=new_version_key)
        _LOGGER.info(
            "Setting Production key : %d ,to this new key = %s", production_key, new_version_key
        )
        serialized_metadata = ""
        if metadata:
            serialized_metadata = dumps(deserialized_object=metadata)
        serialized_artifact = dumps(deserialized_object=artifact)
        pipe.hset(
            name=new_version_key,
            mapping={
                "artifact": serialized_artifact,
                "version": str(version),
                "timestamp": time.time(),
                "metadata": serialized_metadata,
            },
        )
        return new_version_key

    def load(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        latest: bool = True,
        version: str = None,
    ) -> Optional[ArtifactData]:
        """
        Loads the artifact from redis registry. Either latest or version (one of the arguments)
         is needed to load the respective artifact.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            latest: load the model in production stage
            version: version to load

        Returns:
            ArtifactData instance
        """
        if (latest and version) or (not latest and not version):
            raise ValueError("Either One of 'latest' or 'version' needed in load method call")
        key = self.construct_key(skeys, dkeys)
        try:
            if latest:
                artifact_data = self.__load_latest_artifact(key)
                self._save_in_cache(key, artifact_data)
            else:
                artifact_data = self.__load_version_artifact(version, key)
        except RedisError as err:
            raise RedisRegistryError(f"{err.__class__.__name__} raised") from err
        else:
            return artifact_data

    def save(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        artifact: artifact_t,
        **metadata: META_VT,
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
        key = self.construct_key(skeys, dkeys)
        production_key = self.__construct_production_key(key)
        version = 0
        try:
            if self.client.exists(production_key):
                _LOGGER.debug("Production key exists for the model")
                version_key = self.client.get(name=production_key)
                version = int(self.get_version(version_key.decode())) + 1
            with self.client.pipeline() as pipe:
                new_version_key = self.__save_artifact(pipe, artifact, metadata, key, str(version))
                pipe.expire(name=new_version_key, time=self.ttl)
                _LOGGER.info("Model is successfully with the key = %s", new_version_key)
                pipe.execute()
        except RedisError as err:
            raise RedisRegistryError(f"{err.__class__.__name__} raised") from err
        else:
            return str(version)

    def delete(self, skeys: KEYS, dkeys: KEYS, version: str) -> None:
        """
        Deletes the model version from registry.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: model version to delete
        """
        key = self.construct_key(skeys, dkeys)
        del_key = self.__construct_version_key(key, version)
        try:
            if self.client.exists(del_key):
                self.client.delete(del_key)
                _LOGGER.info("Model with the key = %s, deleted successfully", del_key)
            else:
                _LOGGER.debug("Key to delete: %s, Not Found !!!\n Exiting.....", del_key)
                raise ModelKeyNotFound(
                    "Key to delete: %s, Not Found !!!\n Exiting....." % del_key,
                )
        except RedisError as err:
            raise RedisRegistryError(f"{err.__class__.__name__} raised") from err
        else:
            self._clear_cache(del_key)

    @staticmethod
    def is_artifact_stale(artifact_data: ArtifactData, freq_hr: int) -> bool:
        """
        Returns whether the given artifact is stale or not, i.e. if
        more time has elapsed since it was last retrained.
        Args:
            artifact_data: ArtifactData object to look into
            freq_hr: Frequency of retraining in hours

        """
        try:
            artifact_ts = float(artifact_data.extras["timestamp"])
        except (KeyError, TypeError) as err:
            raise RedisRegistryError("Error fetching timestamp information") from err
        stale_ts = (datetime.now() - timedelta(hours=freq_hr)).timestamp()
        return stale_ts > artifact_ts
