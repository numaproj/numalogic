import math
from collections import OrderedDict

import mlflow
import torch
from mlflow.entities import RunData, RunInfo, Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.models.model import ModelInfo
from mlflow.store.entities import PagedList
from sklearn.preprocessing import StandardScaler
from torch import tensor
from mlflow.models import Model

from numalogic.models.autoencoder.variants.vanilla import VanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.registry.mlflow_registry import CompositeModel


def create_model():
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)
    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

    loss_fn = torch.nn.MSELoss(reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=1e-6)

    for t in range(1000):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)
        model.zero_grad()
        loss.backward()
        optim.step()

    return model


def model_sklearn():
    return StdDevThreshold()


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


def mock_log_model_pytorch(*_, **__):
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
        model_uri="runs:/a7c0b376530b40d7b23e6ce2081c899c/model",
        model_uuid="a7c0b376530b40d7b23e6ce2081c899c",
        run_id="a7c0b376530b40d7b23e6ce2081c899c",
        saved_input_example_info=None,
        signature_dict=None,
        utc_time_created="2022-05-23 22:35:59.557372",
        mlflow_version="2.0.1",
        signature=None,
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
        model_uri="runs:/a7c0b376530b40d7b23e6ce2081c899c/model",
        model_uuid="a7c0b376530b40d7b23e6ce2081c899c",
        run_id="a7c0b376530b40d7b23e6ce2081c899c",
        saved_input_example_info=None,
        signature_dict=None,
        utc_time_created="2022-05-23 22:35:59.557372",
        mlflow_version="2.0.1",
        signature=None,
    )


def mock_log_model_pyfunc(*_, **__):
    return ModelInfo(
        artifact_path="model",
        flavors={
            "pyfunc": {"model_data": "data", "pyfunc_version": "1.11.0", "code": None},
            "python_function": {
                "pickle_module_name": "mlflow.pyfunc.pickle_module",
                "loader_module": "mlflow.pyfunc",
                "python_version": "3.8.5",
                "data": "data",
                "env": "conda.yaml",
            },
        },
        model_uri="runs:/a7c0b376530b40d7b23e6ce2081c899c/model",
        model_uuid="a7c0b376530b40d7b23e6ce2081c899c",
        run_id="a7c0b376530b40d7b23e6ce2081c899c",
        saved_input_example_info=None,
        signature_dict=None,
        utc_time_created="2022-05-23 22:35:59.557372",
        mlflow_version="2.0.1",
        signature=None,
    )


def mock_load_model_pyfunc(*_, **__):
    artifact_path = "model"
    flavors = {
        "python_function": {
            "cloudpickle_version": "3.0.0",
            "code": None,
            "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
            "loader_module": "mlflow.pyfunc.model",
            "python_model": "python_model.pkl",
            "python_version": "3.10.14",
            "streamable": False,
        }
    }
    model_size_bytes = 8912
    model_uuid = "ae27ecc166c94c01a4f4dccaf84ca5dc"
    run_id = "7e85a3fa46d44e668c840f3dddc909c3"
    utc_time_created = "2024-09-18 17:12:41.501209"
    model = Model(
        artifact_path=artifact_path,
        flavors=flavors,
        model_size_bytes=model_size_bytes,
        model_uuid=model_uuid,
        run_id=run_id,
        utc_time_created=utc_time_created,
        mlflow_version="2.16.0",
    )
    return mlflow.pyfunc.PyFuncModel(
        model_meta=model,
        model_impl=TestObject(
            python_model=CompositeModel(
                skeys=["error"],
                dict_artifacts={
                    "inference": VanillaAE(10),
                    "precrocessing": StandardScaler(),
                    "threshold": StdDevThreshold(),
                },
                **{"learning_rate": 0.01},
            )
        ),
    )


def mock_load_model_pyfunc_type_error(*_, **__):
    artifact_path = "model"
    flavors = {
        "python_function": {
            "cloudpickle_version": "3.0.0",
            "code": None,
            "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
            "loader_module": "mlflow.pytorch.model",
            "python_model": "python_model.pkl",
            "python_version": "3.10.14",
            "streamable": False,
        }
    }
    model_size_bytes = 8912
    model_uuid = "ae27ecc166c94c01a4f4dccaf84ca5dc"
    run_id = "7e85a3fa46d44e668c840f3dddc909c3"
    utc_time_created = "2024-09-18 17:12:41.501209"
    model = Model(
        artifact_path=artifact_path,
        flavors=flavors,
        model_size_bytes=model_size_bytes,
        model_uuid=model_uuid,
        run_id=run_id,
        utc_time_created=utc_time_created,
        mlflow_version="2.16.0",
    )
    return mlflow.pyfunc.PyFuncModel(
        model_meta=model, model_impl=mlflow.pytorch._PyTorchWrapper(VanillaAE(10), device="cpu")
    )


def mock_transition_stage(*_, **__):
    return ModelVersion(
        creation_timestamp=1653402941169,
        current_stage="Production",
        description="",
        last_updated_timestamp=1653402941191,
        name="test::error",
        run_id="a7c0b376530b40d7b23e6ce2081c899c",
        run_link="",
        source="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
        status="READY",
        status_message="",
        tags={},
        user_id="",
        version="5",
    )


def mock_get_model_version(*_, **__):
    return [
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="test::error",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_link="",
            source="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="5",
        )
    ]


def mock_get_model_version_obj(*_, **__):
    return ModelVersion(
        creation_timestamp=1653402941169,
        current_stage="Production",
        description="",
        last_updated_timestamp=1653402941191,
        name="test::error",
        run_id="a7c0b376530b40d7b23e6ce2081c899c",
        run_link="",
        source="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
        status="READY",
        status_message="",
        tags={},
        user_id="",
        version="5",
    )


def mock_list_of_model_version(*_, **__):
    model_list = [
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="test::error",
            run_id="6e85c26e6e8b49fdb493807d5a527a2c",
            run_link="",
            source="mlflow-artifacts:/0/6e85c26e6e8b49fdb493807d5a527a2c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="8",
        ),
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="test::error",
            run_id="6e85c26e6e8b49fdb493807d5a527a2c",
            run_link="",
            source="mlflow-artifacts:/0/6e85c26e6e8b49fdb493807d5a527a2c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="9",
        ),
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="test::error",
            run_id="6e85c26e6e8b49fdb493807d5a527a2c",
            run_link="",
            source="mlflow-artifacts:/0/6e85c26e6e8b49fdb493807d5a527a2c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="10",
        ),
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="test::error",
            run_id="6e85c26e6e8b49fdb493807d5a527a2c",
            run_link="",
            source="mlflow-artifacts:/0/6e85c26e6e8b49fdb493807d5a527a2c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="11",
        ),
    ]

    return PagedList(items=model_list, token=None)


def mock_list_of_model_version2(*_, **__):
    return PagedList(items=mock_get_model_version(), token=None)


def return_scaler():
    scaler = StandardScaler()
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler.fit_transform(data)
    return scaler


def return_empty_rundata():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            end_time=None,
            experiment_id="0",
            lifecycle_stage="active",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_uuid="a7c0b376530b40d7b23e6ce2081c899c",
            start_time=1658788772612,
            status="RUNNING",
            user_id="lol",
        ),
        run_data=RunData(metrics={}, tags={}, params={}),
    )


def return_sklearn_rundata():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            end_time=None,
            experiment_id="0",
            lifecycle_stage="active",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_uuid="a7c0b376530b40d7b23e6ce2081c899c",
            start_time=1658788772612,
            status="RUNNING",
            user_id="lol",
        ),
        run_data=RunData(metrics={}, tags={}, params={}),
    )


def return_pyfunc_rundata():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            end_time=None,
            experiment_id="0",
            lifecycle_stage="active",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_uuid="a7c0b376530b40d7b23e6ce2081c899c",
            start_time=1658788772612,
            status="RUNNING",
            user_id="lol",
        ),
        run_data=RunData(
            metrics={}, tags={}, params=[mlflow.entities.Param("learning_rate", "0.01")]
        ),
    )


def return_pytorch_rundata_dict():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            end_time=None,
            experiment_id="0",
            lifecycle_stage="active",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_uuid="a7c0b376530b40d7b23e6ce2081c899c",
            start_time=1658788772612,
            status="RUNNING",
            user_id="lol",
        ),
        run_data=RunData(metrics={}, tags={}, params=[mlflow.entities.Param("lr", "0.001")]),
    )


class TestObject(mlflow.pyfunc.PythonModel):
    def __init__(self, python_model):
        self.python_model = python_model
