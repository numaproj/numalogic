import logging
from enum import Enum
from typing import Optional, Sequence

import mlflow.pyfunc
import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from numalogic.registry import ArtifactManager, ArtifactData
from numalogic.tools.types import Artifact

_LOGGER = logging.getLogger()


class ModelStage(str, Enum):
    """
    Defines different stages the model state can be in mlflow
    """

    STAGE = "Staging"
    ARCHIVE = "Archived"
    PRODUCTION = "Production"


class MLflowRegistry(ArtifactManager):
    """
    Model saving and loading using MLFlow Registry.

    More details here: https://mlflow.org/docs/latest/model-registry.html

    Args:
        tracking_uri: the tracking server uri to use for mlflow
        artifact_type: the type of primary artifact to use
                              supported values include:
                              {"pytorch", "sklearn", "tensorflow", "pyfunc"}
        models_to_retain: number of models to retain in the DB (default = 5)

    Examples
    --------
    >>> from numalogic.models.autoencoder.variants.vanilla import VanillaAE
    >>> from numalogic.registry.mlflow_registry import MLflowRegistry
    >>> from sklearn.pipeline import make_pipeline
    >>>
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler.fit(data)
    >>> registry = MLflowRegistry(tracking_uri="http://0.0.0.0:8080", artifact_type="pytorch")
    >>> registry.save(skeys=["model"], dkeys=["AE"], artifact=VanillaAE(10))
    >>> artifact_data = registry.load(skeys=["model"], dkeys=["AE"])
    """

    _TRACKING_URI = None

    def __new__(
        cls,
        tracking_uri: Optional[str],
        artifact_type: str = "pytorch",
        models_to_retain: int = 5,
        *args,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        if (not cls._TRACKING_URI) or (cls._TRACKING_URI != tracking_uri):
            cls._TRACKING_URI = tracking_uri
        return instance

    def __init__(
        self, tracking_uri: str, artifact_type: str = "pytorch", models_to_retain: int = 5
    ):
        super().__init__(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.handler = self.mlflow_handler(artifact_type)
        self.models_to_retain = models_to_retain

    @staticmethod
    def construct_key(skeys: Sequence[str], dkeys: Sequence[str]) -> str:
        """
        Returns a single key comprising static and dynamic key fields.
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
    def mlflow_handler(artifact_type: str):
        """
        Helper method to return the right handler given the artifact type.
        """
        if artifact_type == "pytorch":
            return mlflow.pytorch
        if artifact_type == "sklearn":
            return mlflow.sklearn
        if artifact_type == "tensorflow":
            return mlflow.tensorflow
        if artifact_type == "pyfunc":
            return mlflow.pyfunc
        raise NotImplementedError("Artifact Type not Implemented")

    def load(
        self,
        skeys: Sequence[str],
        dkeys: Sequence[str],
        latest: bool = True,
        version: str = None,
    ) -> Optional[ArtifactData]:
        model_key = self.construct_key(skeys, dkeys)
        try:
            if latest:
                model = self.handler.load_model(
                    model_uri=f"models:/{model_key}/{ModelStage.PRODUCTION}"
                )
                version_info = self.client.get_latest_versions(
                    model_key, stages=[ModelStage.PRODUCTION]
                )[-1]
            elif version is not None:
                model = self.handler.load_model(model_uri=f"models:/{model_key}/{version}")
                version_info = self.client.get_model_version(model_key, version)
            else:
                raise ValueError("One of 'latest' or 'version' needed in load method call")
            _LOGGER.info("Successfully loaded model %s from Mlflow", model_key)

            run_info = mlflow.get_run(version_info.run_id)
            metadata = run_info.data.params or None
            _LOGGER.info("Successfully loaded model metadata from Mlflow!")

            return ArtifactData(artifact=model, metadata=metadata, extras=dict(version_info))
        except Exception as ex:
            _LOGGER.exception("Error when loading a model with key: %s: %r", model_key, ex)
            return None

    def save(
        self,
        skeys: Sequence[str],
        dkeys: Sequence[str],
        artifact: Artifact,
        **metadata: str,
    ) -> Optional[ModelVersion]:
        """
        Saves the artifact into mlflow registry and updates version.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            artifact: primary artifact to be saved
            metadata: additional metadata surrounding the artifact that needs to be saved

        Returns:
            mlflow ModelVersion instance
        """
        model_key = self.construct_key(skeys, dkeys)
        try:
            mlflow.start_run()
            self.handler.log_model(artifact, "model", registered_model_name=model_key)
            if metadata:
                mlflow.log_params(metadata)
            model_version = self.transition_stage(skeys=skeys, dkeys=dkeys)
            _LOGGER.info("Successfully inserted model %s to Mlflow", model_key)
            return model_version
        except Exception as ex:
            _LOGGER.exception("Error when saving a model with key: %s: %r", model_key, ex)
            return None
        finally:
            mlflow.end_run()

    def delete(self, skeys: Sequence[str], dkeys: Sequence[str], version: str) -> None:
        """
        Deletes the artifact with a specified version from mlflow registry.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            version: explicit artifact version

        Returns:
             None
        """
        model_key = self.construct_key(skeys, dkeys)
        try:
            self.client.delete_model_version(name=model_key, version=version)
            _LOGGER.info("Successfully deleted model %s", model_key)
        except Exception as ex:
            _LOGGER.exception("Error when deleting a model with key: %s: %r", model_key, ex)

    def transition_stage(
        self, skeys: Sequence[str], dkeys: Sequence[str]
    ) -> Optional[ModelVersion]:
        """
        Changes stage information for the given model. Sets new model to "Production". The old
        production model is set to "Staging" and the rest model versions are set to "Archived".

        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
        Returns:
             mlflow ModelVersion instance
        """
        model_name = self.construct_key(skeys, dkeys)
        try:
            version = int(self.get_version(model_name=model_name))
            latest_model_data = self.client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage=ModelStage.PRODUCTION,
            )
            if version - 1 > 0:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=str(version - 1),
                    stage=ModelStage.STAGE,
                )
            if version - 2 > 0:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=str(version - 2),
                    stage=ModelStage.ARCHIVE,
                )

            # only keep "models_to_retain" number of models.
            list_model_versions = list(self.client.search_model_versions(f"name='{model_name}'"))
            models_to_delete = list_model_versions[: -self.models_to_retain]
            for stale_model in models_to_delete:
                self.delete(skeys=skeys, dkeys=dkeys, version=stale_model.version)
            _LOGGER.info("Successfully transitioned model to Production stage")
            return latest_model_data
        except Exception as ex:
            _LOGGER.exception(
                "Error when transitioning a model: %s to different stage: %r", model_name, ex
            )
            return None

    def get_version(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get model's latest version given the model name
        Args:
            model_name: model name for which the version has to be identified.
        Returns:
            version from mlflow ModelVersion instance
        """
        try:
            return self.client.get_latest_versions(name=model_name, stages=[])[-1].version
        except RestException as ex:
            _LOGGER.error("Error when getting model version: %r", ex)
            return None
