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
from typing import Any, Optional
from datetime import datetime, timedelta

import boto3
from botocore.exceptions import ClientError

from numalogic.registry import ArtifactManager, ArtifactCache, ArtifactData
from numalogic.tools.exceptions import DynamoDBRegistryError
from numalogic.tools.types import KEYS, artifact_t, META_VT
from numalogic.registry._serialize import loads, dumps

_LOGGER = logging.getLogger(__name__)


class AWSConnector:
    """
    Class to connect to any AWS resource.

    Args:
        role: AWS role
        region: AWS region (default: us-west-2)
    """

    __slots__ = ("role", "region", "session", "sess_start_time")

    __STS_RESET_SECS = 30 * 60

    def __init__(self, role: str, region="us-west-2"):
        self.role = role
        self.region = region
        self.session = None
        self.sess_start_time = None

    def get_session(self, new_session=False) -> boto3.Session:
        if new_session:
            self.session = None

        if (not self.session) or (time.time() - self.sess_start_time > self.__STS_RESET_SECS):
            self.session = self._get_sts_session()
            self.sess_start_time = time.time()
        return self.session

    def _get_sts_session(self):
        _LOGGER.info("Getting session with role: %s and region: %s", self.role, self.region)
        assumed_role_object = boto3.client("sts").assume_role(
            RoleArn=self.role, RoleSessionName="AssumeRoleSession1"
        )
        return boto3.session.Session(
            aws_access_key_id=assumed_role_object["Credentials"]["AccessKeyId"],
            aws_secret_access_key=assumed_role_object["Credentials"]["SecretAccessKey"],
            aws_session_token=assumed_role_object["Credentials"]["SessionToken"],
            region_name=self.region,
        )

    def get_client(self, service="dynamodb", **kwargs):
        return self.get_session(**kwargs).client(service)

    def get_resource(self, service="dynamodb", **kwargs):
        return self.get_session(**kwargs).resource(service)


class DynamoDBRegistry(ArtifactManager):
    """
    Model saving and loading to and from Dynamodb.
    Args:
        table: table name to use
        role: AWS role with access to DynamoDB table
        models_to_retain: number of models to retain in the DB (default = 2)
        cache_registry: ArtifactCache instance, must have an expiration set for the
         model to be refreshed.

    Examples
    --------
    >>> from numalogic.models.autoencoder.variants import VanillaAE
    >>> from numalogic.registry import DynamoDBRegistry
    >>> ...
    >>> registry = DynamoDBRegistry(table="mytable", role="arn:aws:iam::1234567890:role/role-name")
    >>> skeys, dkeys = ("mymetric", "ae"), ("vanilla", "seq10")
    >>> model = VanillaAE(seq_len=10)
    >>> registry.save(skeys, dkeys, artifact=model, **{'lr': 0.01})
    >>> loaded_artifact = registry.load(skeys, dkeys).
    """

    __slots__ = ("table_name", "role", "connector", "models_to_retain", "cache_registry")

    def __init__(
        self,
        table: str,
        role: str,
        models_to_retain: int = 2,
        cache_registry: Optional[ArtifactCache] = None,
    ):
        super().__init__(table)
        self.table_name = table
        self.role = role
        self.connector = AWSConnector(role=role)
        self.models_to_retain = models_to_retain
        self.cache_registry = cache_registry

    def create_table(self) -> dict[str, Any]:
        """
        Creates a new table with a specific schema.

        Returns
        -------
            Table instance.
        """
        return self.connector.get_client().create_table(
            TableName=self.table_name,
            KeySchema=[
                {"AttributeName": "skey", "KeyType": "HASH"},  # partition key
                {"AttributeName": "dkey", "KeyType": "RANGE"},  # sort key
            ],
            LocalSecondaryIndexes=[
                {
                    "IndexName": "TimeSearch-IndexLocal",
                    "KeySchema": [
                        {"AttributeName": "skey", "KeyType": "HASH"},
                        {
                            "AttributeName": "dkey",
                            "KeyType": "RANGE",
                        },
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            AttributeDefinitions=[
                {"AttributeName": "skey", "AttributeType": "S"},
                {"AttributeName": "dkey", "AttributeType": "S"},
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )

    @staticmethod
    def _get_cache_key(part_key: str, v_sort_key: str) -> str:
        return f"{part_key}::{v_sort_key}"

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

    @staticmethod
    def _unpack_artifact_data(
        item: dict[str, Any],
    ) -> ArtifactData:
        serialized_artifact = item.get("artifact")
        artifact_version = item.get("version")
        artifact_timestamp = item.get("timestamp")
        serialized_metadata = item.get("metadata")

        serialized_artifact = (
            serialized_artifact.value
            if hasattr(serialized_artifact, "value")
            else serialized_artifact
        )
        deserialized_artifact = loads(serialized_artifact)
        deserialized_metadata = None
        if serialized_metadata:
            serialized_metadata = (
                serialized_metadata.value
                if hasattr(serialized_metadata, "value")
                else serialized_metadata
            )
            deserialized_metadata = loads(serialized_metadata).get("metadata")

        return ArtifactData(
            artifact=deserialized_artifact,
            metadata=deserialized_metadata,
            extras={
                "timestamp": float(artifact_timestamp),
                "version": artifact_version,
            },
        )

    @staticmethod
    def _pack_artifact_data(
        artifact: artifact_t,
        version: str,
        **metadata: META_VT,
    ) -> dict[str, Any]:
        serialized_artifact = dumps(artifact)
        serialized_metadata = ""
        if metadata:
            serialized_metadata = dumps(deserialized_object=metadata)

        return {
            "artifact": serialized_artifact,
            "metadata": serialized_metadata,
            "version": str(version),
            "timestamp": str(time.time()),
        }

    def load(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        latest: bool = True,
        version: Optional[str] = None,
    ) -> Optional[ArtifactData]:
        """
        Loads the desired artifact and metadata from dynamodb.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            latest: ignored for now
            version: ignored for now
        Returns:
            ArtifactData instance.
        """
        if (latest and version) or (not latest and not version):
            raise ValueError("Either One of 'latest' or 'version' needed in load method call")

        table = self.connector.get_resource().Table(self.table_name)
        part_key = self.construct_key(skeys)
        sort_key = self.construct_key(dkeys)

        if latest:
            v_sort_key = self._construct_version_key(sort_key)
        else:
            # Version is provided
            v_sort_key = self._construct_version_key(sort_key, version=version)

        cache_key = self._get_cache_key(part_key, v_sort_key)
        cached_artifact = self._load_from_cache(cache_key)

        if cached_artifact:
            _LOGGER.debug("Found cached artifact for key: %s", cache_key)
            return cached_artifact

        try:
            response = table.get_item(Key={"skey": part_key, "dkey": v_sort_key})
        except ClientError as err:
            raise DynamoDBRegistryError(f"{err.__class__.__name__} raised") from err

        item = response.get("Item") or {}

        if item and "data" in item:
            artifact_data = self._unpack_artifact_data(item.get("data"))
            self._save_in_cache(cache_key, artifact_data)
            return artifact_data

        _LOGGER.info("Record not found for skey: %s, dkey: %s", part_key, sort_key)
        return None

    @staticmethod
    def __save_item(
        table, part_key: str, sort_key: str, data: dict[str, Any], version: Optional[int] = None
    ):
        try:
            response = table.put_item(
                Item={"skey": part_key, "dkey": sort_key, "data": data, "version": version}
            )
        except ClientError as err:
            raise DynamoDBRegistryError(f"{err.__class__.__name__} raised") from err
        return response

    def save(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        artifact: artifact_t,
        **metadata: META_VT,
    ) -> Optional[str]:
        r"""
        Saves the artifact into dynamodb.
        Note that this overwrites any previously
        saved item with the same skey and dkey.
        Args:
            artifact:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            artifact: model to be saved
            metadata: additional metadata surrounding the artifact that needs to be saved.
        """
        table = self.connector.get_resource().Table(self.table_name)
        part_key = self.construct_key(skeys)
        sort_key = self.construct_key(dkeys)

        # Check if model exists
        artifact_info = self.load(skeys, dkeys, latest=True)

        if artifact_info:
            version = int(artifact_info.extras.get("version")) + 1
        else:
            version = 1

        artifact_data = self._pack_artifact_data(artifact, str(version), **metadata)

        # Save a copy of the artifact with the current version.
        self.__save_item(
            table, part_key, self._construct_version_key(sort_key, version=version), artifact_data
        )

        # Save the zeroth version (v0) as the one pointing to the current version.
        self.__save_item(
            table, part_key, self._construct_version_key(sort_key), artifact_data, version=version
        )

        # Delete the stale model
        stale_version = version - self.models_to_retain
        if stale_version > 0:
            _LOGGER.info("Deleting stale version %s", stale_version)
            self.__delete_item(
                table, part_key, self._construct_version_key(sort_key, version=stale_version)
            )

        return str(version)

    @staticmethod
    def __delete_item(table, part_key: str, v_sort_key: str):
        try:
            response = table.delete_item(Key={"skey": part_key, "dkey": v_sort_key})
        except ClientError as err:
            raise DynamoDBRegistryError(f"{err.__class__.__name__} raised") from err
        return response

    def delete(self, skeys: KEYS, dkeys: KEYS, version: str) -> Optional[dict[str, Any]]:
        """
        Deletes the artifact from dynamodb.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: the version of the artifact to be deleted.
        """
        if version == "0":
            raise ValueError("Cannot delete the main (zeroth) record directly!")

        table = self.connector.get_resource().Table(self.table_name)
        part_key = self.construct_key(skeys)
        sort_key = self.construct_key(dkeys)

        artifact_data = self.load(skeys, dkeys, latest=True)

        if artifact_data:
            # Check if the latest version is getting deleted
            if version == artifact_data.extras.get("version"):
                # Find if an older version exists
                prev_version = str(int(version) - 1)
                artifact_data_prev = self.load(skeys, dkeys, latest=False, version=prev_version)
                if artifact_data_prev and prev_version != "0":
                    # Update the zeroth record pointer to the previous version
                    prev_item = self._pack_artifact_data(
                        artifact_data_prev.artifact, prev_version, **artifact_data_prev.metadata
                    )
                    self.__save_item(
                        table,
                        part_key,
                        self._construct_version_key(sort_key),
                        prev_item,
                        version=prev_version,
                    )
                else:
                    # Delete the zeroth record pointer if no previous version exists
                    v_sort_key = self._construct_version_key(sort_key)
                    self.__delete_item(table, part_key, v_sort_key)
                    self._clear_cache(self._get_cache_key(part_key, v_sort_key))

        # Delete the main version record
        v_sort_key = self._construct_version_key(sort_key, version)
        response = self.__delete_item(table, part_key, v_sort_key)
        self._clear_cache(self._get_cache_key(part_key, v_sort_key))
        return response

    @staticmethod
    def _construct_version_key(sort_key: str, version: int = 0) -> str:
        return f"v{version}__{sort_key}"

    @staticmethod
    def construct_key(keys: KEYS) -> str:
        """
        Returns a single composite key from a list of static or dynamic key elements.
        Args:
            keys: key fields as list/tuple of strings (static or dynamic).

        Returns
        -------
            Joined key.
        """
        return ":".join(keys)

    @staticmethod
    def is_artifact_stale(artifact_data: ArtifactData, freq_hr: int) -> bool:
        """Returns whether the given artifact is stale or not, i.e. if
        more time has elapsed since it was last retrained.

        Args:
        ----
            artifact_data: ArtifactData object to look into
            freq_hr: Frequency of retraining in hours.

        """
        try:
            artifact_ts = float(artifact_data.extras["timestamp"])
        except (KeyError, TypeError) as err:
            raise DynamoDBRegistryError("Error fetching timestamp information") from err
        stale_ts = (datetime.now() - timedelta(hours=freq_hr)).timestamp()
        return stale_ts > artifact_ts
