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
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any

import mlflow.pyfunc
import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from torch import nn

from numalogic.registry import ArtifactManager, ArtifactData
from numalogic.registry.artifact import ArtifactCache
from numalogic.tools.exceptions import ModelVersionError
from numalogic.tools.types import artifact_t, KEYS, META_VT

_LOGGER = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Defines different stages the model state can be in mlflow."""

    STAGE = "Staging"
    ARCHIVE = "Archived"
    PRODUCTION = "Production"


class MLflowRegistry(ArtifactManager):
    """Model saving and loading using MLFlow Registry. The parameter model_stage
    determines what environment we are using. The old models are moved to
    'Archived' state and the latest model comes to 'Staging' or 'Production'
    depending on model_stage parameter.

    More details here: https://mlflow.org/docs/latest/model-registry.html

    Args:
    ----
        tracking_uri: the tracking server uri to use for mlflow
        models_to_retain: number of models to retain in the DB (default = 5)
        model_stage: Staging environment from where to load the latest model from (mlflow )
                            supported values include:
                              {"Staging", "Production", "Archived"}(default = "Production")

    Examples
    --------
    >>> from numalogic.models.autoencoder.variants import VanillaAE
    >>> from numalogic.registry import MLflowRegistry
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler.fit(data)
    >>> registry = MLflowRegistry(tracking_uri="http://0.0.0.0:8080")
    >>> registry.save(skeys=["model"], dkeys=["AE"], artifact=VanillaAE(10))
    >>> artifact_data = registry.load(skeys=["model"], dkeys=["AE"], artifact_type="pytorch")
    """

    __slots__ = ("client", "models_to_retain", "model_stage", "cache_registry")
    _TRACKING_URI = None

    def __new__(
        cls,
        tracking_uri: Optional[str],
        models_to_retain: int = 5,
        model_stage: ModelStage = ModelStage.PRODUCTION,
        cache_registry: Optional[ArtifactCache] = None,
        *args,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        if (not cls._TRACKING_URI) or (cls._TRACKING_URI != tracking_uri):
            cls._TRACKING_URI = tracking_uri
        return instance

    def __init__(
        self,
        tracking_uri: str,
        models_to_retain: int = 5,
        model_stage: str = ModelStage.PRODUCTION,
        cache_registry: Optional[ArtifactCache] = None,
    ):
        super().__init__(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.models_to_retain = models_to_retain
        self.model_stage = model_stage
        self.cache_registry = cache_registry

    @staticmethod
    def handler_from_obj(artifact: artifact_t):
        if isinstance(artifact, nn.Module):
            return mlflow.pytorch
        if isinstance(artifact, BaseEstimator):
            return mlflow.sklearn
        return mlflow.pyfunc

    @staticmethod
    def handler_from_type(artifact_type: str):
        """Helper method to return the right handler given the artifact type."""
        if artifact_type == "pytorch":
            return mlflow.pytorch
        if artifact_type == "sklearn":
            return mlflow.sklearn
        if artifact_type == "pyfunc":
            return mlflow.pyfunc
        raise NotImplementedError("Artifact Type not Implemented")

    def _load_from_cache(self, key: str) -> Optional[ArtifactData]:
        if not self.cache_registry:
            return None
        return self.cache_registry.load(key)

    def _save_in_cache(self, key: str, artifact_data: ArtifactData) -> None:
        if self.cache_registry:
            self.cache_registry.save(key, artifact_data)

    def _clear_cache(self, key: str) -> Optional[ArtifactData]:
        if self.cache_registry:
            return self.cache_registry.delete(key)
        return None

    def load(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        latest: bool = True,
        version: Optional[str] = None,
        artifact_type: str = "pytorch",
    ) -> Optional[ArtifactData]:
        """Load the artifact from the registry. The artifact is loaded from the cache if available.

        Args:
        ----
            skeys: Static keys
            dkeys: Dynamic keys
            latest: Load the latest version of the model (default = True)
            version: Version of the model to load (default = None)
            artifact_type: Type of the artifact to load (default = "pytorch").

        Returns
        -------
            The loaded ArtifactData object if available otherwise None
        """
        model_key = self.construct_key(skeys, dkeys)

        if (latest and version) or (not latest and not version):
            raise ValueError("Either One of 'latest' or 'version' needed in load method call")

        try:
            if latest:
                cached_artifact = self._load_from_cache(model_key)
                if cached_artifact:
                    _LOGGER.debug("Found cached artifact for key: %s", model_key)
                    return cached_artifact
                version_info = self.client.get_latest_versions(model_key, stages=[self.model_stage])
                if not version_info:
                    raise ModelVersionError("Model version missing for key = %s" % model_key)
                version_info = version_info[-1]
            else:
                version_info = self.client.get_model_version(model_key, version)
            model, metadata = self.__load_artifacts(skeys, dkeys, version_info, artifact_type)
        except RestException as mlflow_err:
            return self.__log_mlflow_err(mlflow_err, model_key)
        except ModelVersionError:
            _LOGGER.exception("No Model found found in model stage: %s", self.model_stage)
            return None
        except Exception:
            _LOGGER.exception("Unexpected error while Registry loading with key: %s", model_key)
            return None
        else:
            artifact_data = ArtifactData(
                artifact=model, metadata=metadata, extras=dict(version_info)
            )
            # save in cache if loading the latest version
            if latest:
                self._save_in_cache(model_key, artifact_data)
            return artifact_data

    @staticmethod
    def __log_mlflow_err(mlflow_err: RestException, model_key: str) -> None:
        if ErrorCode.Value(mlflow_err.error_code) == RESOURCE_DOES_NOT_EXIST:
            _LOGGER.info("Model not found with key: %s", model_key)
        else:
            _LOGGER.exception(
                "Mlflow error when loading a model with key: %s: %r", model_key, mlflow_err
            )

    def save(
        self,
        skeys: KEYS,
        dkeys: KEYS,
        artifact: artifact_t,
        run_id: Optional[str] = None,
        **metadata: META_VT,
    ) -> Optional[ModelVersion]:
        """Saves the artifact into mlflow registry and updates version.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            artifact: primary artifact to be saved
            run_id: mlflow run id
            metadata: additional metadata surrounding the artifact that needs to be saved.

        Returns
        -------
            mlflow ModelVersion instance
        """
        model_key = self.construct_key(skeys, dkeys)
        handler = self.handler_from_obj(artifact)
        try:
            mlflow.start_run(run_id=run_id)
            handler.log_model(artifact, "model", registered_model_name=model_key)
            if metadata:
                mlflow.log_params(metadata)
            model_version = self.transition_stage(skeys=skeys, dkeys=dkeys)
        except Exception:
            _LOGGER.exception("Unhandled error when saving a model with key: %s", model_key)
            return None
        else:
            _LOGGER.info("Successfully inserted model %s to Mlflow", model_key)
            return model_version
        finally:
            mlflow.end_run()

    @staticmethod
    def is_artifact_stale(artifact_data: ArtifactData, freq_hr: int) -> bool:
        """Returns whether the given artifact is stale or not, i.e. if
        more time has elasped since it was last retrained.

        Args:
        ----
            artifact_data: ArtifactData object to look into
            freq_hr: Frequency of retraining in hours.

        """
        date_updated = artifact_data.extras["last_updated_timestamp"] / 1000
        stale_date = (datetime.now() - timedelta(hours=freq_hr)).timestamp()
        return date_updated < stale_date

    def delete(self, skeys: KEYS, dkeys: KEYS, version: str) -> None:
        """Deletes the artifact with a specified version from mlflow registry.

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: explicit artifact version.

        Returns
        -------
             None
        """
        model_key = self.construct_key(skeys, dkeys)
        try:
            self.client.delete_model_version(name=model_key, version=version)
            _LOGGER.info("Successfully deleted model %s", model_key)
        except Exception:
            _LOGGER.exception("Error when deleting a model with key: %s", model_key)
        else:
            self._clear_cache(model_key)

    def transition_stage(self, skeys: KEYS, dkeys: KEYS) -> Optional[ModelVersion]:
        """Changes stage information for the given model. Sets new model to "Production". The old
        production model is set to "Staging" and the rest model versions are set to "Archived".

        Args:
        ----
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
        Returns:
             mlflow ModelVersion instance
        """
        model_name = self.construct_key(skeys, dkeys)
        try:
            current_staging = self.client.get_latest_versions(
                name=model_name, stages=[self.model_stage]
            )
            latest = self.client.get_latest_versions(name=model_name, stages=["None"])

            latest_model_data = self.client.transition_model_version_stage(
                name=model_name, version=str(latest[-1].version), stage=self.model_stage
            )

            if current_staging:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=str(current_staging[-1].version),
                    stage=ModelStage.ARCHIVE,
                )

            # only keep "models_to_retain" number of models.
            self.__delete_stale_models(skeys=skeys, dkeys=dkeys)
        except RestException:
            _LOGGER.exception("Error when transitioning a model: %s to different stage", model_name)
            return None
        else:
            _LOGGER.info("Successfully transitioned model to Production stage")
            return latest_model_data

    def __delete_stale_models(self, skeys: KEYS, dkeys: KEYS):
        model_name = self.construct_key(skeys, dkeys)
        list_model_versions = list(self.client.search_model_versions(f"name='{model_name}'"))
        if len(list_model_versions) > self.models_to_retain:
            models_to_delete = list_model_versions[self.models_to_retain :]
            for stale_model in models_to_delete:
                self.delete(skeys=skeys, dkeys=dkeys, version=stale_model.version)
                _LOGGER.debug("Deleted stale model version : %s", stale_model.version)

    def __load_artifacts(
        self, skeys: KEYS, dkeys: KEYS, version_info: ModelVersion, artifact_type: str
    ) -> tuple[artifact_t, dict[str, Any]]:
        model_key = self.construct_key(skeys, dkeys)
        handler = self.handler_from_type(artifact_type)
        model = handler.load_model(model_uri=f"models:/{model_key}/{version_info.version}")
        _LOGGER.info("Successfully loaded model %s from Mlflow", model_key)

        run_info = mlflow.get_run(version_info.run_id)
        metadata = run_info.data.params or {}
        _LOGGER.info(
            "Successfully loaded model = %s with version %s Mlflow!",
            model_key,
            version_info.version,
        )
        return model, metadata
