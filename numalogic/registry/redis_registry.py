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

import logging
import time
from datetime import datetime, timedelta
from typing import Optional
import orjson
import redis.client

from redis.exceptions import RedisError

from numalogic.registry.artifact import ArtifactManager, ArtifactData, ArtifactCache, _apply_jitter
from numalogic.registry._serialize import loads, dumps
from numalogic.tools.exceptions import ModelKeyNotFound, RedisRegistryError
from numalogic.tools.types import artifact_t, redis_client_t, KEYS, META_T, META_VT, KeyedArtifact

_LOGGER = logging.getLogger(__name__)


class RedisRegistry(ArtifactManager):
    """Model saving and loading using Redis Registry.

    Args:
    ----
        client: Take in the redis client already established/created
        ttl: Total Time to Live (in seconds) for the key when saving in redis (dafault = 604800)
        jitter_sec: Jitter (in secs) added to model timestamp information to solve
                    Thundering Herd problem (default = 30 mins)
        jitter_steps_sec: Step interval value (in sec) for jitter_sec value (default = 120 secs)
        cache_registry: Cache registry to use (default = None).
        transactional: Flag to indicate if the registry should be transactional or
        not (default = False).

    Examples
    --------
    >>> import redis
    >>> from numalogic.models.autoencoder.variants import VanillaAE
    >>> from numalogic.registry.redis_registry import RedisRegistry
    >>> ...
    >>> r = redis.Redis(host='127.0.0.1', port=6379)
    >>> registry = RedisRegistry(client=r)
    >>> skeys, dkeys = ("mymetric", "ae"), ("vanilla", "seq10")
    >>> model = VanillaAE(seq_len=10)
    >>> registry.save(skeys, dkeys, artifact=model, **{'lr': 0.01})
    >>> loaded_artifact = registry.load(skeys, dkeys)
    """

    __slots__ = (
        "client",
        "ttl",
        "jitter_sec",
        "jitter_steps_sec",
        "cache_registry",
        "transactional",
    )

    def __init__(
        self,
        client: redis_client_t,
        ttl: int = 604800,
        jitter_sec: int = 30 * 60,
        jitter_steps_sec: int = 2 * 60,
        cache_registry: Optional[ArtifactCache] = None,
        transactional: bool = True,
    ):
        super().__init__("")
        self.client = client
        self.ttl = ttl
        self.jitter_sec = jitter_sec
        self.jitter_steps_sec = jitter_steps_sec
        self.cache_registry = cache_registry
        self.transactional = transactional

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

    @staticmethod
    def __construct_latest_key(key: str):
        return RedisRegistry.construct_key(skeys=[key], dkeys=["LATEST"])

    @staticmethod
    def __construct_version_key(key: str, version: str):
        return RedisRegistry.construct_key(skeys=[key], dkeys=[version])

    @staticmethod
    def get_version(key: str) -> str:
        """Get version number from the string
        Args:
            key: full model key.

        Returns
        -------
            version
        """
        return key.split("::")[-1]

    def _load_from_cache(self, key: str) -> Optional[ArtifactData]:
        if not self.cache_registry:
            return None
        return self.cache_registry.load(key)

    def _save_in_cache(self, key: str, artifact_data: ArtifactData) -> None:
        if self.cache_registry:
            _LOGGER.debug("Saving artifact in cache with key: %s", key)
            self.cache_registry.save(key, artifact_data)

    def _clear_cache(self, key: Optional[str] = None) -> Optional[ArtifactData]:
        if self.cache_registry:
            return self.cache_registry.delete(key) if key else self.cache_registry.clear()
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
            deserialized_metadata = orjson.loads(serialized_metadata)
        return ArtifactData(
            artifact=deserialized_artifact,
            metadata=deserialized_metadata,
            extras={
                "timestamp": float(artifact_timestamp.decode()),
                "version": artifact_version.decode(),
                "source": self._STORETYPE,
            },
        )

    def __load_latest_artifact(self, key: str) -> tuple[ArtifactData, bool]:
        """
        Load the latest artifact from the registry.

        Args:
            key: full model key.

        Returns
        -------
            ArtifactData and a boolean flag indicating if the artifact was loaded from cache.

        Raises
        ------
            ModelKeyNotFound: If the model key is not found in the registry.
        """
        latest_key = self.__construct_latest_key(key)
        cached_artifact = self._load_from_cache(latest_key)
        if cached_artifact:
            _LOGGER.debug("Found cached artifact for key: %s", latest_key)
            return cached_artifact, True
        if not self.client.exists(latest_key):
            raise ModelKeyNotFound(f"latest key: {latest_key}, Not Found !!!")
        model_key = self.client.get(latest_key)
        _LOGGER.debug("latest key, %s, is pointing to the key : %s", latest_key, model_key)
        artifact, _ = self.__load_version_artifact(
            version=self.get_version(model_key.decode()), key=key
        )
        return artifact, False

    def __load_version_artifact(self, version: str, key: str) -> tuple[ArtifactData, bool]:
        version_key = self.__construct_version_key(key, version)
        cached_artifact = self._load_from_cache(version_key)
        if cached_artifact:
            _LOGGER.debug("Found cached version artifact for key: %s", version_key)
            return cached_artifact, True
        if not self.client.exists(version_key):
            raise ModelKeyNotFound(f"Could not find model key with key: {version_key}")
        return (
            self.__get_artifact_data(
                model_key=version_key,
            ),
            False,
        )

    def __save_artifact(
        self, pipe, artifact: artifact_t, key: KEYS, version: str, **metadata: META_T
    ) -> str:
        new_version_key = self.__construct_version_key(key, version)
        latest_key = self.__construct_latest_key(key)
        pipe.set(name=latest_key, value=new_version_key)
        _LOGGER.debug("Setting latest key : %s ,to this new key = %s", latest_key, new_version_key)
        serialized_metadata = orjson.dumps(metadata) if metadata else b""
        serialized_artifact = dumps(deserialized_object=artifact)
        _cur_ts = int(time.time())
        pipe.hset(
            name=new_version_key,
            mapping={
                "artifact": serialized_artifact,
                "version": version,
                "timestamp": _apply_jitter(_cur_ts, self.jitter_sec, self.jitter_steps_sec),
                "metadata": serialized_metadata,
            },
        )
        return new_version_key

    def load(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        latest: bool = True,
        version: Optional[str] = None,
    ) -> Optional[ArtifactData]:
        """Loads the artifact from redis registry. Either latest or version (one of the arguments)
         is needed to load the respective artifact.

         If cache registry is provided, it will first check the cache registry for the artifact.
         If latest is passed, latest key is saved otherwise version call saves the respective
         version artifact in cache.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            latest: load the model in latest stage
            version: version to load.

        Returns
        -------
            ArtifactData instance

        Raises
        ------
            ValueError: If both latest and version are provided or none of them are provided.
            RedisRegistryError: If any redis error occurs.
        """
        if (latest and version) or (not latest and not version):
            raise ValueError("Either One of 'latest' or 'version' needed in load method call")
        key = self.construct_key(skeys, dkeys)
        try:
            if latest:
                artifact_data, is_cached = self.__load_latest_artifact(key)
            else:
                artifact_data, is_cached = self.__load_version_artifact(version, key)
        except RedisError as err:
            raise RedisRegistryError(f"{err.__class__.__name__} raised") from err
        else:
            if not is_cached:
                if latest:
                    _LOGGER.debug(
                        "Saving %s, in cache as %s", self.__construct_latest_key(key), key
                    )
                    self._save_in_cache(self.__construct_latest_key(key), artifact_data)
                else:
                    _LOGGER.info(
                        "Saving %s,  in cache as %s",
                        self.__construct_version_key(key, version),
                        key,
                    )
                    self._save_in_cache(self.__construct_version_key(key, version), artifact_data)
            return artifact_data

    def save(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        artifact: artifact_t,
        _pipe: Optional[redis.client.Pipeline] = None,
        **metadata: META_VT,
    ) -> Optional[str]:
        """Saves the artifact into redis registry and updates version.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            artifact: primary artifact to be saved
            _pipe: RedisPipeline object
            metadata: additional metadata surrounding the artifact that needs to be saved.

        Returns
        -------
            Model version (str)

        Raises
        ------
            RedisRegistryError: If there is any RedisError while saving the artifact.
        """
        key = self.construct_key(skeys, dkeys)
        latest_key = self.__construct_latest_key(key)
        version = 0
        try:
            if self.client.exists(latest_key):
                _LOGGER.debug("Latest key: %s exists for the model", latest_key)
                version_key = self.client.get(name=latest_key)
                version = int(self.get_version(version_key.decode())) + 1
            _redis_pipe = (
                self.client.pipeline(transaction=self.transactional) if _pipe is None else _pipe
            )
            new_version_key = self.__save_artifact(
                pipe=_redis_pipe, artifact=artifact, key=key, version=str(version), **metadata
            )
            _redis_pipe.expire(name=new_version_key, time=self.ttl)
            if _pipe is None:
                _redis_pipe.execute()
        except RedisError as err:
            raise RedisRegistryError(f"{err.__class__.__name__} raised") from err
        else:
            _LOGGER.info("Model with the key = %s, saved successfully.", new_version_key)
            return str(version)

    def delete(self, skeys: KEYS, dkeys: KEYS, version: str) -> None:
        """Deletes the model version from registry.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: model version to delete.

        Raises
        ------
            ModelKeyNotFound: If the model version is not found in registry.
            RedisRegistryError: If there is any RedisError while deleting the artifact.
        """
        key = self.construct_key(skeys, dkeys)
        del_key = self.__construct_version_key(key, version)
        try:
            if self.client.exists(del_key):
                self.client.delete(del_key)
            else:
                raise ModelKeyNotFound(
                    f"Key to delete: {del_key}, Not Found!",
                )
        except RedisError as err:
            raise RedisRegistryError(f"{err.__class__.__name__} raised") from err
        else:
            _LOGGER.info("Model with the key = %s, deleted successfully", del_key)
            self._clear_cache(del_key)

    @staticmethod
    def is_artifact_stale(artifact_data: ArtifactData, freq_hr: int) -> bool:
        """Returns whether the given artifact is stale or not, i.e. if
        more time has elapsed since it was last retrained.

        Args:
        ----
            artifact_data: ArtifactData object to look into
            freq_hr: Frequency of retraining in hours.

        Returns
        -------
            True if artifact is stale, False otherwise.

        Raises
        ------
            RedisRegistryError: If there is any error while fetching timestamp information.
        """
        try:
            artifact_ts = float(artifact_data.extras["timestamp"])
        except (KeyError, TypeError) as err:
            raise RedisRegistryError("Error fetching timestamp information") from err
        stale_ts = (datetime.now() - timedelta(hours=freq_hr)).timestamp()
        return stale_ts > artifact_ts

    def __update_metadata(self, skeys: KEYS, dict_artifacts: dict[str, KeyedArtifact], metadata):
        try:
            with self.client.pipeline(transaction=self.transactional) as pipe:
                pipe.multi()
                for _, value in dict_artifacts.items():
                    key = self.construct_key(skeys, value.dkeys)
                    latest_key = self.__construct_latest_key(key)
                    version_key = self.client.get(name=latest_key)
                    pipe.hset(
                        name=version_key.decode(), key="metadata", value=orjson.dumps(metadata)
                    )
                pipe.execute()
        except RedisError as err:
            raise RedisRegistryError(f"{err.__class__.__name__} raised") from err

    def save_multiple(
        self,
        skeys: KEYS,
        dict_artifacts: dict[str, KeyedArtifact],
        **metadata: META_VT,
    ):
        """
        Saves multiple artifacts into redis registry. The last save stores all the
        artifact versions in the metadata.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dict_artifacts: dict of artifacts to save
            metadata: additional metadata surrounding the artifact that needs to be saved.
        """
        dict_model_ver = {}
        try:
            for value in dict_artifacts.values():
                dict_model_ver[":".join(value.dkeys)] = self.save(
                    skeys=skeys,
                    dkeys=value.dkeys,
                    artifact=value.artifact,
                    **metadata,
                )
            self.__update_metadata(
                skeys=skeys,
                dict_artifacts=dict_artifacts,
                metadata={**{"artifact_versions": dict_model_ver}, **metadata},
            )
            _LOGGER.info("Successfully saved all the artifacts with: %s", dict_model_ver)
        except RedisError as err:
            raise RedisRegistryError(f"{err.__class__.__name__} raised") from err
        else:
            return dict_model_ver
