import math

import mlflow
import torch
from mlflow.entities import RunData, RunInfo, Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.models.model import ModelInfo
from mlflow.store.entities import PagedList
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


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
    return {"lr": 0.01}


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
        mlflow_version="1.26.0",
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
        model_uri="runs:/f2dad48d86c748358b47bdaa24b2619c/model",
        model_uuid="adisajdasjdoasd",
        run_id="a7c0b376530b40d7b23e6ce2081c899c",
        saved_input_example_info=None,
        signature_dict=None,
        utc_time_created="2022-05-23 22:35:59.557372",
        mlflow_version="1.26.0",
        signature=None,
    )


def mock_transition_stage(*_, **__):
    return ModelVersion(
        creation_timestamp=1653402941169,
        current_stage="Production",
        description="",
        last_updated_timestamp=1653402941191,
        name="testtest:error",
        run_id="a7c0b376530b40d7b23e6ce2081c899c",
        run_link="",
        source="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
        status="READY",
        status_message="",
        tags={},
        user_id="",
        version="5",
    )


def return_empty_rundata():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts",
            end_time=1653402941290,
            experiment_id="0",
            lifecycle_stage="active",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_uuid="a7c0b376530b40d7b23e6ce2081c899c",
            start_time=1653402941169,
            status="READY",
            user_id="lol",
        ),
        run_data=RunData(
            metrics={},
            params={},
            tags=[
                mlflow.entities.RunTag("run_id", "a7c0b376530b40d7b23e6ce2081c899c"),
                mlflow.entities.RunTag("artifact_path", "model"),
                mlflow.entities.RunTag("utc_time_created", "2022-11-24 18:26:51.414513"),
                mlflow.entities.RunTag("model_uuid", "a7c0b376530b40d7b23e6ce2081c899c"),
                mlflow.entities.RunTag("mlflow_version", "2.0.1"),
            ],
        ),
    )


def mock_get_latest_model_version(*_, **__):
    return [
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="model_nnet:error",
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


def mock_get_model_version(*_, **__):
    return ModelVersion(
        creation_timestamp=1653402941169,
        current_stage="Production",
        description="",
        last_updated_timestamp=1653402941191,
        name="model_nnet:error",
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
            name="testtest:error",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_link="",
            source="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="5",
        ),
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="testtest:error",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_link="",
            source="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="6",
        ),
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="testtest:error",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_link="",
            source="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="7",
        ),
        ModelVersion(
            creation_timestamp=1653402941169,
            current_stage="Production",
            description="",
            last_updated_timestamp=1653402941191,
            name="testtest:error",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_link="",
            source="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="8",
        ),
    ]

    return PagedList(items=model_list, token=None)


def mock_list_of_model_version2(*_, **__):
    return PagedList(items=mock_get_latest_model_version(), token=None)


def return_scaler():
    scaler = StandardScaler()
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler.fit_transform(data)
    return scaler


def return_sklearn_rundata():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts",
            end_time=1669315807932,
            experiment_id="0",
            lifecycle_stage="active",
            run_name="",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_uuid="a7c0b376530b40d7b23e6ce2081c899c",
            start_time=1658788772612,
            status="RUNNING",
            user_id="lol",
        ),
        run_data=RunData(metrics={}, tags={}, params={}),
    )


def return_pytorch_rundata_dict():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts",
            end_time=1669315807932,
            experiment_id="0",
            lifecycle_stage="active",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_uuid="a7c0b376530b40d7b23e6ce2081c899c",
            start_time=1658788772612,
            status="FINISHED",
            user_id="lol",
        ),
        run_data=RunData(
            metrics={}, tags={}, params=[mlflow.entities.Param(key="lr", value="0.01")]
        ),
    )


def return_pytorch_rundata_dict_no_metadata():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts",
            end_time=1669315807932,
            experiment_id="0",
            lifecycle_stage="active",
            run_id="a7c0b376530b40d7b23e6ce2081c899c",
            run_uuid="a7c0b376530b40d7b23e6ce2081c899c",
            start_time=1658788772612,
            status="FINISHED",
            user_id="lol",
        ),
        run_data=RunData(metrics={}, tags={}, params={}),
    )
