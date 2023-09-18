import unittest
from contextlib import contextmanager
from unittest.mock import patch, Mock

from freezegun import freeze_time
from mlflow import ActiveRun
from mlflow.exceptions import RestException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode, RESOURCE_LIMIT_EXCEEDED
from mlflow.store.entities import PagedList
from sklearn.preprocessing import StandardScaler

from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import MLflowRegistry, ArtifactData, LocalLRUCache
from numalogic.registry.mlflow_registry import ModelStage
from numalogic.tools.exceptions import ModelVersionError
from tests.registry._mlflow_utils import (
    model_sklearn,
    create_model,
    mock_log_model_pytorch,
    mock_log_state_dict,
    mock_get_model_version,
    mock_transition_stage,
    mock_log_model_sklearn,
    return_pytorch_rundata_dict,
    return_empty_rundata,
    mock_list_of_model_version,
    mock_list_of_model_version2,
    return_sklearn_rundata,
    mock_get_model_version_obj,
)

TRACKING_URI = "http://0.0.0.0:5009"


class TestMLflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = create_model()
        cls.model_sklearn = model_sklearn()
        cls.skeys = ["test"]
        cls.dkeys = ["error"]

    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type as err:
            raise self.failureException(f"{exc_type.__name__} raised") from err

    def test_construct_key(self):
        skeys = ["model_", "nnet"]
        dkeys = ["error1"]
        key = MLflowRegistry.construct_key(skeys, dkeys)
        self.assertEqual("model_:nnet::error1", key)

    @patch("mlflow.pytorch.log_model", mock_log_model_pytorch)
    @patch("mlflow.log_param", mock_log_state_dict)
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version)
    def test_save_model(self):
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        status = ml.save(skeys=skeys, dkeys=dkeys, artifact=self.model, run_id="1234")
        mock_status = "READY"
        self.assertEqual(mock_status, status.status)

    @patch("mlflow.sklearn.log_model", mock_log_model_sklearn)
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_sklearn_rundata())))
    @patch("mlflow.active_run", Mock(return_value=return_sklearn_rundata()))
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    def test_save_model_sklearn(self):
        model = self.model_sklearn
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        status = ml.save(skeys=skeys, dkeys=dkeys, artifact=model)
        mock_status = "READY"
        self.assertEqual(mock_status, status.status)

    @patch("mlflow.pytorch.log_model", mock_log_model_pytorch())
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch("mlflow.log_params", {"lr": 0.01})
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.pytorch.load_model", Mock(return_value=VanillaAE(10)))
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_pytorch_rundata_dict()))
    def test_load_model_when_pytorch_model_exist1(self):
        model = self.model
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        ml.save(skeys=skeys, dkeys=dkeys, artifact=model, **{"lr": 0.01})
        data = ml.load(skeys=skeys, dkeys=dkeys, artifact_type="pytorch")
        self.assertIsNotNone(data.metadata)
        self.assertIsInstance(data.artifact, VanillaAE)

    @patch("mlflow.pytorch.log_model", mock_log_model_pytorch())
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.pytorch.load_model", Mock(return_value=VanillaAE(10)))
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_empty_rundata()))
    def test_load_model_when_pytorch_model_exist2(self):
        model = self.model
        ml = MLflowRegistry(TRACKING_URI, models_to_retain=2)
        skeys = self.skeys
        dkeys = self.dkeys
        ml.save(skeys=skeys, dkeys=dkeys, artifact=model)
        data = ml.load(skeys=skeys, dkeys=dkeys, artifact_type="pytorch")
        self.assertEqual(data.metadata, {})
        self.assertIsInstance(data.artifact, VanillaAE)

    @patch("mlflow.sklearn.log_model", mock_log_model_sklearn)
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_sklearn_rundata())))
    @patch("mlflow.active_run", Mock(return_value=return_sklearn_rundata()))
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_empty_rundata()))
    @patch.object(
        MLflowRegistry,
        "load",
        Mock(
            return_value=ArtifactData(
                artifact=StandardScaler(), extras={"metric": ["error"]}, metadata={}
            )
        ),
    )
    def test_load_model_when_sklearn_model_exist(self):
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        scaler = StandardScaler()
        ml.save(skeys=skeys, dkeys=dkeys, artifact=scaler)
        data = ml.load(skeys=skeys, dkeys=dkeys)
        print(data)
        self.assertIsInstance(data.artifact, StandardScaler)
        self.assertEqual(data.metadata, {})

    @patch("mlflow.pytorch.log_model", mock_log_model_pytorch())
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_empty_rundata())))
    @patch("mlflow.active_run", Mock(return_value=return_empty_rundata()))
    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_model_version", mock_get_model_version_obj)
    @patch("mlflow.pytorch.load_model", Mock(return_value=VanillaAE(10)))
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_empty_rundata()))
    def test_load_model_with_version(self):
        model = self.model
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        ml.save(skeys=skeys, dkeys=dkeys, artifact=model)
        data = ml.load(skeys=skeys, dkeys=dkeys, version="5", latest=False)
        self.assertIsInstance(data.artifact, VanillaAE)
        self.assertEqual(data.metadata, {})

    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch(
        "mlflow.tracking.MlflowClient.get_latest_versions",
        Mock(return_value=PagedList(items=[], token=None)),
    )
    @patch("mlflow.pytorch.load_model", Mock(return_value=VanillaAE(10)))
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_empty_rundata()))
    def test_staging_model_load_error(self):
        ml = MLflowRegistry(TRACKING_URI, model_stage=ModelStage.STAGE)
        skeys = self.skeys
        dkeys = self.dkeys
        ml.load(skeys=skeys, dkeys=dkeys)
        self.assertRaises(ModelVersionError)

    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version())
    @patch("mlflow.pytorch.load_model", Mock(return_value=VanillaAE(10)))
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_empty_rundata()))
    def test_both_version_latest_model_with_version(self):
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        with self.assertRaises(ValueError):
            ml.load(skeys=skeys, dkeys=dkeys, latest=False)

    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version())
    @patch("mlflow.pyfunc.load_model", Mock(side_effect=RuntimeError))
    def test_load_model_when_no_model_01(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]
        ml = MLflowRegistry(TRACKING_URI)
        with self.assertLogs(level="ERROR") as log:
            ml.load(skeys=fake_skeys, dkeys=fake_dkeys, artifact_type="pyfunc")
            self.assertTrue(log.output)

    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version())
    @patch("mlflow.pytorch.load_model", Mock(side_effect=RuntimeError))
    def test_load_model_when_no_model_02(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]
        ml = MLflowRegistry(TRACKING_URI)
        with self.assertLogs(level="ERROR") as log:
            ml.load(skeys=fake_skeys, dkeys=fake_dkeys, artifact_type="pytorch")
            self.assertTrue(log.output)

    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch(
        "mlflow.tracking.MlflowClient.transition_model_version_stage",
        Mock(side_effect=RestException({"error_code": ErrorCode.Name(RESOURCE_LIMIT_EXCEEDED)})),
    )
    def test_transition_stage_fail(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]
        ml = MLflowRegistry(TRACKING_URI)
        with self.assertLogs(level="ERROR") as log:
            ml.transition_stage(fake_skeys, fake_dkeys)
            self.assertTrue(log.output)

    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version())
    def test_no_implementation(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]
        ml = MLflowRegistry(TRACKING_URI)
        with self.assertLogs(level="ERROR") as log:
            ml.load(skeys=fake_skeys, dkeys=fake_dkeys, artifact_type="somerandom")
            self.assertTrue(log.output)

    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch("mlflow.pytorch.log_model", mock_log_model_pytorch)
    @patch("mlflow.log_params", mock_log_state_dict)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.tracking.MlflowClient.search_model_versions", mock_list_of_model_version2)
    @patch("mlflow.tracking.MlflowClient.delete_model_version", None)
    @patch("mlflow.pytorch.load_model", Mock(side_effect=RuntimeError))
    def test_delete_model_when_model_exist(self):
        model = self.model
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        ml.save(skeys=skeys, dkeys=dkeys, artifact=model, **{"lr": 0.01})
        ml.delete(skeys=skeys, dkeys=dkeys, version="5")
        with self.assertLogs(level="ERROR") as log:
            ml.load(skeys=skeys, dkeys=dkeys)
            self.assertTrue(log.output)

    @patch("mlflow.tracking.MlflowClient.delete_model_version", Mock(side_effect=RuntimeError))
    def test_delete_model_when_no_model(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]
        ml = MLflowRegistry(TRACKING_URI)
        with self.assertLogs(level="ERROR") as log:
            ml.delete(skeys=fake_skeys, dkeys=fake_dkeys, version="1")
            self.assertTrue(log.output)

    @patch("mlflow.pytorch.log_model", Mock(side_effect=RuntimeError))
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_empty_rundata())))
    @patch("mlflow.active_run", Mock(return_value=return_empty_rundata()))
    def test_save_failed(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]

        ml = MLflowRegistry(TRACKING_URI)
        with self.assertLogs(level="ERROR") as log:
            ml.save(skeys=fake_skeys, dkeys=fake_dkeys, artifact=self.model)
            self.assertTrue(log.output)

    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch(
        "mlflow.tracking.MlflowClient.get_latest_versions",
        Mock(side_effect=RestException({"error_code": ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)})),
    )
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_pytorch_rundata_dict()))
    def test_load_no_model_found(self):
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        data = ml.load(skeys=skeys, dkeys=dkeys, artifact_type="pytorch")
        self.assertIsNone(data)

    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch(
        "mlflow.tracking.MlflowClient.get_latest_versions",
        Mock(side_effect=RestException({"error_code": ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)})),
    )
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_pytorch_rundata_dict()))
    def test_load_other_mlflow_err(self):
        ml = MLflowRegistry(TRACKING_URI)
        skeys = self.skeys
        dkeys = self.dkeys
        self.assertIsNone(ml.load(skeys=skeys, dkeys=dkeys, artifact_type="pytorch"))

    @patch("mlflow.pytorch.log_model", mock_log_model_pytorch())
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch("mlflow.log_params", {"lr": 0.01})
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.pytorch.load_model", Mock(return_value=VanillaAE(10)))
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_pytorch_rundata_dict()))
    def test_is_model_stale_true(self):
        model = self.model
        ml = MLflowRegistry(TRACKING_URI)
        ml.save(skeys=self.skeys, dkeys=self.dkeys, artifact=model, **{"lr": 0.01})
        data = ml.load(skeys=self.skeys, dkeys=self.dkeys, artifact_type="pytorch")
        self.assertTrue(ml.is_artifact_stale(data, 12))

    @patch("mlflow.pytorch.log_model", mock_log_model_pytorch())
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch("mlflow.log_params", {"lr": 0.01})
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.pytorch.load_model", Mock(return_value=VanillaAE(10)))
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_pytorch_rundata_dict()))
    def test_is_model_stale_false(self):
        model = self.model
        ml = MLflowRegistry(TRACKING_URI)
        ml.save(skeys=self.skeys, dkeys=self.dkeys, artifact=model, **{"lr": 0.01})
        data = ml.load(skeys=self.skeys, dkeys=self.dkeys, artifact_type="pytorch")
        with freeze_time("2022-05-24 10:30:00"):
            self.assertFalse(ml.is_artifact_stale(data, 12))

    def test_no_cache(self):
        registry = MLflowRegistry(TRACKING_URI)
        self.assertIsNone(
            registry._save_in_cache(
                "key", ArtifactData(artifact=self.model, extras={}, metadata={})
            )
        )
        self.assertIsNone(registry._load_from_cache("key"))
        self.assertIsNone(registry._clear_cache("key"))

    def test_cache(self):
        cache_registry = LocalLRUCache()
        registry = MLflowRegistry(TRACKING_URI, cache_registry=cache_registry)
        registry._save_in_cache("key", ArtifactData(artifact=self.model, extras={}, metadata={}))
        self.assertIsNotNone(registry._load_from_cache("key"))
        self.assertIsNotNone(registry._clear_cache("key"))

    @patch("mlflow.pytorch.log_model", mock_log_model_pytorch())
    @patch("mlflow.start_run", Mock(return_value=ActiveRun(return_pytorch_rundata_dict())))
    @patch("mlflow.active_run", Mock(return_value=return_pytorch_rundata_dict()))
    @patch("mlflow.log_params", {"lr": 0.01})
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.pytorch.load_model", Mock(return_value=VanillaAE(10)))
    @patch("mlflow.tracking.MlflowClient.get_run", Mock(return_value=return_pytorch_rundata_dict()))
    def test_cache_loading(self):
        cache_registry = LocalLRUCache(ttl=50000)
        ml = MLflowRegistry(TRACKING_URI, cache_registry=cache_registry)
        ml.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.model, **{"lr": 0.01})
        ml.load(skeys=self.skeys, dkeys=self.dkeys, artifact_type="pytorch")
        key = MLflowRegistry.construct_key(self.skeys, self.dkeys)
        self.assertIsNotNone(ml._load_from_cache(key))
        data = ml.load(skeys=self.skeys, dkeys=self.dkeys, artifact_type="pytorch")
        self.assertIsNotNone(data)


if __name__ == "__main__":
    unittest.main()
