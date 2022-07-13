import math
import unittest
from collections import OrderedDict
from contextlib import contextmanager
from unittest import mock
from unittest.mock import patch, Mock

import mlflow
import mlflow.pytorch
import mlflow.pytorch
import torch
from mlflow.entities.model_registry import ModelVersion
from mlflow.models.model import ModelInfo
from sklearn.ensemble import RandomForestRegressor
from torch import tensor

from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import MLflowRegistrar

TRACKING_URI = "http://0.0.0.0:5009"


def create_model():
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)
    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

    loss_fn = torch.nn.MSELoss(reduction="sum")

    learning_rate = 1e-6
    for t in range(1000):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    return model


def model_sklearn():
    params = {"n_estimators": 5, "random_state": 42}
    sk_learn_rfr = RandomForestRegressor(**params)
    return sk_learn_rfr


def mock_log_state_dict(*_, **__):
    return OrderedDict(
        [
            (
                "encoder.0.weight",
                tensor(
                    [
                        [
                            0.2635,
                            0.5033,
                            -0.2808,
                            -0.4609,
                            0.2749,
                            -0.5048,
                            -0.0960,
                            0.6310,
                            -0.4750,
                            0.1700,
                        ],
                        [
                            -0.1626,
                            0.1635,
                            -0.2873,
                            0.5045,
                            -0.3312,
                            0.0791,
                            -0.4530,
                            -0.5068,
                            0.1734,
                            0.0485,
                        ],
                        [
                            -0.5209,
                            -0.1975,
                            -0.3471,
                            -0.6511,
                            0.5214,
                            0.4137,
                            -0.2795,
                            0.2267,
                            0.2497,
                            0.3451,
                        ],
                    ]
                ),
            )
        ]
    )


def mock_log_model(*_, **__):
    return ModelInfo(
        artifact_path="model",
        flavors={
            "pytorch": {"model_data": "data", "pytorch_version": "1.11.0", "code": None},
            "python_function": {
                "pickle_module_name": "mlflow.pytorch.pickle_module",
                "loader_module": "mlflow.pytorch",
                "python_version": "3.8.5",
                "data": "data",
                "env": "conda.yaml",
            },
        },
        model_uri="runs:/f2dad48d86c748358b47bdaa24b2619c/model",
        model_uuid="adisajdasjdoasd",
        run_id="f2dad48d86c748358b47bdaa24b2619c",
        saved_input_example_info=None,
        signature_dict=None,
        utc_time_created="2022-05-23 22:35:59.557372",
        mlflow_version="1.26.0",
    )


def mock_log_model_sklearn(*_, **__):
    return ModelInfo(
        artifact_path="model",
        flavors={
            "sklearn": {"model_data": "data", "sklearn_version": "1.11.0", "code": None},
            "python_function": {
                "pickle_module_name": "mlflow.sklearn.pickle_module",
                "loader_module": "mlflow.sklearn",
                "python_version": "3.8.5",
                "data": "data",
                "env": "conda.yaml",
            },
        },
        model_uri="runs:/f2dad48d86c748358b47bdaa24b2619c/model",
        model_uuid="adisajdasjdoasd",
        run_id="f2dad48d86c748358b47bdaa24b2619c",
        saved_input_example_info=None,
        signature_dict=None,
        utc_time_created="2022-05-23 22:35:59.557372",
        mlflow_version="1.26.0",
    )


def mock_transition_stage(*_, **__):
    return ModelVersion(
        creation_timestamp=1653402941169,
        current_stage="Production",
        description="",
        last_updated_timestamp=1653402941191,
        name="testtest:error",
        run_id="6e85c26e6e8b49fdb493807d5a527a2c",
        run_link="",
        source="mlflow-artifacts:/0/6e85c26e6e8b49fdb493807d5a527a2c/artifacts/model",
        status="READY",
        status_message="",
        tags={},
        user_id="",
        version="5",
    )


def mock_start_run(*_, **__):
    return mlflow.tracking.fluent.ActiveRun


def mock_get_model_version(*_, **__):
    return [
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="testtest:error",
            run_id="6e85c26e6e8b49fdb493807d5a527a2c",
            run_link="",
            source="mlflow-artifacts:/0/6e85c26e6e8b49fdb493807d5a527a2c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="5",
        )
    ]


def mock_load_model(*_, **__):
    return mlflow.pytorch.PyFuncModel


class TestMLflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = create_model()
        cls.model_sklearn = model_sklearn()

    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException("{} raised".format(exc_type.__name__))

    def test_construct_key(self):
        skeys = ["model_", "nnet"]
        dkeys = ["error1"]
        key = MLflowRegistrar.construct_key(skeys, dkeys)
        self.assertEqual("model_:nnet::error1", key)

    @patch("mlflow.pytorch.log_model", mock_log_model)
    @patch("mlflow.log_param", mock_log_state_dict)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.start_run", mock_start_run)
    def test_insert_model(self):
        model = self.model
        ml = MLflowRegistrar(TRACKING_URI, artifact_type="pytorch")
        skeys = ["model_", "nnet"]
        dkeys = ["error1"]
        status = ml.save(
            skeys=skeys, dkeys=dkeys, artifact=model, artifact_state_dict=model.state_dict()
        )
        mock_status = "READY"
        self.assertEqual(mock_status, status.status)

    @patch("mlflow.sklearn.log_model", mock_log_model_sklearn)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.start_run", mock_start_run)
    def test_insert_model_sklearn(self):
        model = self.model_sklearn
        ml = MLflowRegistrar(TRACKING_URI, artifact_type="sklearn")
        skeys = ["model_", "nnet"]
        dkeys = ["error1"]
        status = ml.save(skeys=skeys, dkeys=dkeys, artifact=model)
        mock_status = "READY"
        self.assertEqual(mock_status, status.status)

    @patch("mlflow.pytorch.log_model", VanillaAE(10))
    @patch("mlflow.log_param", OrderedDict({"a": 1}))
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    def test_select_model_when_model_exist(self):
        model = self.model
        ml = MLflowRegistrar(TRACKING_URI, artifact_type="pytorch")
        skeys = ["model_", "nnet"]
        dkeys = ["error1"]
        ml.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=model,
        )
        ml.load = mock.Mock(return_value=(VanillaAE(10), None))
        model, metadata = ml.load(skeys=skeys, dkeys=dkeys)
        self.assertEqual(type(model), VanillaAE)

    @patch("mlflow.sklearn.log_model", mock_log_model_sklearn)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    def test_select_model_when_sklearn_model_exist(self):
        model = self.model_sklearn
        ml = MLflowRegistrar(TRACKING_URI, artifact_type="sklearn")
        skeys = ["model_", "nnet"]
        dkeys = ["error1"]
        ml.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=model,
        )
        ml.load = mock.Mock(return_value=(model_sklearn(), None))
        model, state_dict = ml.load(skeys=skeys, dkeys=dkeys)
        self.assertEqual(type(model), RandomForestRegressor)
        self.assertEqual(state_dict, None)

    @patch("mlflow.pytorch.log_model", mock_log_model)
    @patch("mlflow.log_param", OrderedDict({"a": 1}))
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.pytorch.load_model", mock_load_model)
    def test_select_model_with_version(self):
        model = self.model
        ml = MLflowRegistrar(TRACKING_URI)
        skeys = ["model_", "nnet"]
        dkeys = ["error1"]
        ml.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=model,
        )
        ml.load = mock.Mock(return_value=(VanillaAE(10), None))
        model, state_dict = ml.load(skeys=skeys, dkeys=dkeys, version="1", latest=False)
        self.assertEqual(type(model), VanillaAE)
        self.assertEqual(state_dict, None)

    @patch("mlflow.pyfunc.log_model", mock_log_model)
    @patch("mlflow.log_param", mock_log_state_dict)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.pyfunc.load_model", Mock(side_effect=RuntimeError))
    def test_select_model_when_no_model_01(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]
        ml = MLflowRegistrar(TRACKING_URI, artifact_type="pyfunc")
        with self.assertLogs(level="ERROR") as log:
            ml.load(skeys=fake_skeys, dkeys=fake_dkeys)
            self.assertTrue(log.output)

    @patch("mlflow.tensorflow.log_model", mock_log_model)
    @patch("mlflow.log_param", mock_log_state_dict)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.tensorflow.load_model", Mock(side_effect=RuntimeError))
    def test_select_model_when_no_model_02(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]
        ml = MLflowRegistrar(TRACKING_URI, artifact_type="tensorflow")
        with self.assertLogs(level="ERROR") as log:
            ml.load(skeys=fake_skeys, dkeys=fake_dkeys)
            self.assertTrue(log.output)

    def test_no_implementation(self):
        with self.assertRaises(NotImplementedError):
            MLflowRegistrar(TRACKING_URI, artifact_type="some_random")

    @patch("mlflow.pytorch.log_model", mock_log_model)
    @patch("mlflow.log_param", mock_log_state_dict)
    @patch("mlflow.tracking.MlflowClient.transition_model_version_stage", mock_transition_stage)
    @patch("mlflow.tracking.MlflowClient.get_latest_versions", mock_get_model_version)
    @patch("mlflow.tracking.MlflowClient.delete_model_version", None)
    @patch("mlflow.pytorch.load_model", Mock(side_effect=RuntimeError))
    @patch("mlflow.start_run", mock_start_run)
    def test_delete_model_when_model_exist(self):
        model = self.model
        ml = MLflowRegistrar(TRACKING_URI)
        skeys = ["model_", "nnet"]
        dkeys = ["error1"]
        ml.save(skeys=skeys, dkeys=dkeys, artifact=model, **model.state_dict())
        ml.delete(skeys=skeys, dkeys=dkeys, version="1")
        with self.assertLogs(level="ERROR") as log:
            ml.load(skeys=skeys, dkeys=dkeys)
            self.assertTrue(log.output)

    @patch("mlflow.tracking.MlflowClient.delete_model_version", Mock(side_effect=RuntimeError))
    def test_delete_model_when_no_model(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]
        ml = MLflowRegistrar(TRACKING_URI)
        with self.assertLogs(level="ERROR") as log:
            ml.delete(skeys=fake_skeys, dkeys=fake_dkeys, version="1")
            self.assertTrue(log.output)

    @patch("mlflow.pytorch.log_model", Mock(side_effect=RuntimeError))
    def test_insertion_failed(self):
        fake_skeys = ["Fakemodel_"]
        fake_dkeys = ["error"]

        ml = MLflowRegistrar(TRACKING_URI)
        with self.assertLogs(level="ERROR") as log:
            ml.save(skeys=fake_skeys, dkeys=fake_dkeys, artifact=self.model)
            self.assertTrue(log.output)


if __name__ == "__main__":
    unittest.main()
