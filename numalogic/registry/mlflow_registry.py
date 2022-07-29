import codecs
import logging
import pickle
from enum import Enum
from typing import Optional, Sequence, Union, Dict

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

    Args:
        tracking_uri: the tracking server uri to use for mlflow
        artifact_type: the type of primary artifact to use
                              supported values include:
                              {"pytorch", "sklearn", "tensorflow", "pyfunc"}

    Examples
    --------
    >>> from numalogic.models.autoencoder.variants.vanilla import VanillaAE
    >>> from numalogic.preprocess.transformer import LogTransformer
    >>> from numalogic.registry.mlflow_registry import MLflowRegistrar
    >>> from sklearn.preprocessing import StandardScaler, Normalizer
    >>> from sklearn.pipeline import make_pipeline
    >>>
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler.fit(data)
    >>> ml = MLflowRegistrar(tracking_uri= "localhost:8080", artifact_type="pytorch")
    >>> ml.save(skeys=["model"],dkeys=["AE"],primary_artifact=VanillaAE(10),
    secondary_artifacts={"preproc": make_pipeline(scaler)})
    >>> data = ml.load(skeys=["model"],dkeys=["AE"])
    """

    def __init__(self, tracking_uri: str, artifact_type: str = "pytorch"):
        super().__init__(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.handler = self.mlflow_handler(artifact_type)

    @staticmethod
    def __as_dict(
        primary_artifact: Optional[Artifact],
        secondary_artifacts: Union[Sequence[Artifact], Dict[str, Artifact], None],
        metadata: Optional[dict],
        model_properties: Optional[ModelVersion],
    ) -> ArtifactDict:
        """
        Returns a dictionary comprising information on model, metadata, model_properties
        Args:
            primary_artifact: main artifact to be saved
            secondary_artifacts: secondary artifact to be saved
            metadata: ML models metadata
            model_properties: ML model properties (information like time "model_created",
                                    "model_updated_time", "model_name", "tags" , "current stage",
                                    "version"  etc.)

        Returns: ArtifactDict type object
        """
        return {
            "primary_artifact": primary_artifact,
            "secondary_artifacts": secondary_artifacts,
            "metadata": metadata,
            "model_properties": model_properties,
        }

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
        self, skeys: Sequence[str], dkeys: Sequence[str], latest: bool = True, version: str = None
    ) -> ArtifactDict:
        """
        Loads the desired artifact from mlflow registry and returns it.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            latest: boolean field to determine if latest version is desired or not
            version: explicit artifact version

        Returns:
             A dictionary containing primary_artifact, secondary_artifacts, metadata and
             model_properties
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
            secondary_artifacts = None
            model_properties = self.client.get_latest_versions(model_key, stages=["Production"])[-1]
            if model_properties.run_id:
                run_id = model_properties.run_id
                run_data = self.client.get_run(run_id).data.to_dictionary()
                if run_data["params"]:
                    data = run_data["params"]
                    secondary_artifacts = pickle.loads(
                        codecs.decode(data["secondary_artifacts"].encode(), "base64")
                    )
                    _LOGGER.info("Successfully loaded secondary_artifacts from Mlflow")
                    metadata = pickle.loads(codecs.decode(data["metadata"].encode(), "base64"))
                    _LOGGER.info("Successfully loaded model metadata from Mlflow")
            return self.__as_dict(model, secondary_artifacts, metadata, model_properties)
        except Exception as ex:
            _LOGGER.exception("Error when loading a model with key: %s: %r", model_key, ex)
            return {}

    def save(
        self,
        skeys: Sequence[str],
        dkeys: Sequence[str],
        primary_artifact: Artifact,
        secondary_artifacts: Union[Sequence[Artifact], Dict[str, Artifact], None] = None,
        **metadata,
    ) -> Optional[ModelVersion]:
        """
        Saves the artifact into mlflow registry and updates version.
        Args:
            skeys: static key fields as list/tuple of strings
            dkeys: dynamic key fields as list/tuple of strings
            primary_artifact: primary artifact to be saved
            secondary_artifacts: secondary artifact to be saved
            metadata: additional metadata surrounding the artifact that needs to be saved

        Returns:
            mlflow ModelVersion instance
        """
        model_key = self.construct_key(skeys, dkeys)
        try:
            self.handler.log_model(primary_artifact, "model", registered_model_name=model_key)
            if secondary_artifacts:
                secondary_artifacts_data = codecs.encode(
                    pickle.dumps(secondary_artifacts), "base64"
                ).decode()
                mlflow.log_param(key="secondary_artifacts", value=secondary_artifacts_data)
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

    def transition_stage(self, model_name: str) -> Optional[ModelVersion]:
        """
        Changes stage information for the given model. Sets new model to "Production". The old
        production model is set to "Staging" and the rest model versions are set to "Archived".

        Args:
            model_name: model name for which we are updating the stage information.

        Returns:
             mlflow ModelVersion instance
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
        Args:
            model_name: model name for which the version has to be identified.
        Returns:
            version from mlflow ModelVersion instance
        """
        try:
            return self.client.get_latest_versions(model_name, stages=[])[-1].version
        except RestException as ex:
            _LOGGER.error("Error when getting model version: %r", ex)
            return None
