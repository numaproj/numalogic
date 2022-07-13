import codecs
import logging
import pickle
from enum import Enum
from typing import Optional, Sequence

import mlflow.pyfunc
import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from numalogic.registry import ArtifactManager
from numalogic.tools.types import Artifact, ArtifactDict

_LOGGER = logging.getLogger()


class ModelStage(str, Enum):

    """
    Defines different stages the model state can be in mlflow
    """

    STAGE = "Staging"
    ARCHIVE = "Archived"
    PRODUCTION = "Production"


class MLflowRegistrar(ArtifactManager):
    """
    Model saving and loading using MLFlow Registry.

    More details here: https://mlflow.org/docs/latest/model-registry.html

    :param tracking_uri: the tracking server uri to use for mlflow
    :param artifact_type: the type of artifact to use
                          supported values include:
                          {"pytorch", "sklearn", "tensorflow", "pyfunc"}
    """

    def __init__(self, tracking_uri: str, artifact_type: str = "pytorch"):
        super().__init__(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.handler = self.mlflow_handler(artifact_type)

    @staticmethod
    def __as_dict(
        artifact: Optional[Artifact],
        metadata: Optional[dict],
        model_properties: Optional[ModelVersion],
    ) -> ArtifactDict:
        """
        Returns a dictionary comprising information on model, metadata, model_properties

        :param artifact: artifact to be saved
        :param metadata: ML models metadata
        :param model_properties: ML model properties (information like time "model_created",
                                "model_updated_time", "model_name", "tags" , "current stage",
                                "version"  etc.)

        :return: ArtifactDict type object
        """
        return {"artifact": artifact, "metadata": metadata, "model_properties": model_properties}

    @staticmethod
    def construct_key(skeys: Sequence[str], dkeys: Sequence[str]) -> str:
        """
        Returns a single key comprising static and dynamic key fields.

        :param skeys: static key fields as list/tuple of strings
        :param dkeys: dynamic key fields as list/tuple of strings

        :return: key
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
        self, skeys: Sequence[str], dkeys: Sequence[str], latest: bool = True, version: str = None
    ) -> ArtifactDict:
        """
        Loads the desired artifact from mlflow registry and returns it.

        :param skeys: static key fields as list/tuple of strings
        :param dkeys: dynamic key fields as list/tuple of strings
        :param latest: boolean field to determine if latest version is desired or not
        :param version: explicit artifact version

        :return: A tuple of artifact and its metadata
        """

        model_key = self.construct_key(skeys, dkeys)
        try:
            if latest:
                stage = "Production"
                model = self.handler.load_model(model_uri=f"models:/{model_key}/{stage}")
            elif version is not None:
                model = self.handler.load_model(model_uri=f"models:/{model_key}/{version}")
            else:
                _LOGGER.warning("Version not provided in the load mlflow model function call")
                return {}
            _LOGGER.info("Successfully loaded model %s from Mlflow", model_key)
            metadata = None
            model_properties = self.client.get_latest_versions(model_key, stages=["Production"])[-1]
            if model_properties.run_id:
                run_id = model_properties.run_id
                run_data = self.client.get_run(run_id).data.to_dictionary()
                if run_data["params"]:
                    data = run_data["params"]
                    metadata = pickle.loads(codecs.decode(data["metadata"].encode(), "base64"))
                    _LOGGER.info("Successfully loaded model metadata from Mlflow")
            return self.__as_dict(model, metadata, model_properties)
        except Exception as ex:
            _LOGGER.exception("Error when loading a model with key: %s: %r", model_key, ex)
            return {}

    def save(
        self, skeys: Sequence[str], dkeys: Sequence[str], artifact: Artifact, **metadata
    ) -> Optional[ModelVersion]:
        """
        Saves the artifact into mlflow registry and updates version.

        :param skeys: static key fields as list/tuple of strings
        :param dkeys: dynamic key fields as list/tuple of strings
        :param artifact: artifact to be saved
        :param metadata: additional metadata surrounding the artifact that needs to be saved

        :return: mlflow ModelVersion instance
        """
        model_key = self.construct_key(skeys, dkeys)
        try:
            self.handler.log_model(artifact, "model", registered_model_name=model_key)
            if metadata:
                data = codecs.encode(pickle.dumps(metadata), "base64").decode()
                mlflow.log_param(key="metadata", value=data)
                mlflow.log_param(key="model_key", value=model_key)
            model_version = self.transition_stage(model_name=model_key)
            _LOGGER.info("Successfully inserted model %s to Mlflow", model_key)
            return model_version
        except Exception as ex:
            _LOGGER.exception("Error when saving a model with key: %s: %r", model_key, ex)
            return None

    def delete(self, skeys: Sequence[str], dkeys: Sequence[str], version: str) -> None:
        """
        Deletes the artifact with a specified version from mlflow registry.

        :param skeys: static key fields as list/tuple of strings
        :param dkeys: dynamic key fields as list/tuple of strings
        :param version: explicit artifact version

        :return: None
        """
        model_key = self.construct_key(skeys, dkeys)
        try:
            self.client.delete_model_version(name=model_key, version=version)
            _LOGGER.info("Successfully deleted model %s", model_key)
        except Exception as ex:
            _LOGGER.exception("Error when deleting a model with key: %s: %r", model_key, ex)

    def transition_stage(self, model_name: str) -> Optional[ModelVersion]:
        """
        Changes stage information for the given model. Sets new model to "Production". The old production model is set
        to "Staging" and the rest model versions are set to "Archived".

        :param model_name: model name for which we are updating the stage information.

        :return: mlflow ModelVersion instance
        """
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

        :param model_name: model name for which the version has to be identified.


        :return: version from mlflow ModelVersion instance
        """
        try:
            return self.client.get_latest_versions(model_name, stages=[])[-1].version
        except RestException as ex:
            _LOGGER.error("Error when getting model version: %r", ex)
            return None
