import math
from collections import OrderedDict

import mlflow
import torch
from mlflow.entities import RunData, RunInfo, Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.models.model import ModelInfo
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch import tensor


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


def return_scaler():
    scaler = StandardScaler()
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler.fit_transform(data)
    return scaler


def return_empty_rundata():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts",
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
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts",
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
            metrics={},
            tags={},
            params=[
                mlflow.entities.Param(
                    "secondary_artifacts",
                    "gASVRQIAAAAAAACMEHNrbGVhcm4ucGlwZWxpbmWUjAhQaXBlbGluZZSTlCmBlH2UKIwFc3RlcHOU\n"
                    "XZSMDnN0YW5kYXJkc2NhbGVylIwbc2tsZWFybi5wcmVwcm9jZXNzaW5nLl9kYXRhlIwOU3RhbmRh\n"
                    "cmRTY2FsZXKUk5QpgZR9lCiMCXdpdGhfbWVhbpSIjAh3aXRoX3N0ZJSIjARjb3B5lIiMDm5fZmVh\n"
                    "dHVyZXNfaW5flEsCjA9uX3NhbXBsZXNfc2Vlbl+UjBVudW1weS5jb3JlLm11bHRpYXJyYXmUjAZz\n"
                    "Y2FsYXKUk5SMBW51bXB5lIwFZHR5cGWUk5SMAmk4lImIh5RSlChLA4wBPJROTk5K/////0r/////\n"
                    "SwB0lGJDCAQAAAAAAAAAlIaUUpSMBW1lYW5flGgSjAxfcmVjb25zdHJ1Y3SUk5RoFYwHbmRhcnJh\n"
                    "eZSTlEsAhZRDAWKUh5RSlChLAUsChZRoF4wCZjiUiYiHlFKUKEsDaBtOTk5K/////0r/////SwB0\n"
                    "lGKJQxAAAAAAAADgPwAAAAAAAOA/lHSUYowEdmFyX5RoImgkSwCFlGgmh5RSlChLAUsChZRoLIlD\n"
                    "EAAAAAAAANA/AAAAAAAA0D+UdJRijAZzY2FsZV+UaCJoJEsAhZRoJoeUUpQoSwFLAoWUaCyJQxAA\n"
                    "AAAAAADgPwAAAAAAAOA/lHSUYowQX3NrbGVhcm5fdmVyc2lvbpSMBTEuMS4xlHVihpRhjAZtZW1v\n"
                    "cnmUTowHdmVyYm9zZZSJaD5oP3ViLg==\n",
                )
            ],
        ),
    )


def return_pytorch_rundata():
    return Run(
        run_info=RunInfo(
            artifact_uri="mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts",
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
            metrics={},
            tags={},
            params=[
                mlflow.entities.Param(
                    "metadata",
                    "gASV+2AAAAAAAAB9lCiMEG1vZGVsX3N0YXRlX2RpY3SUjAtjb2xsZWN0aW9uc5SMC09yZGVyZWRE\n"
                    "aWN0lJOUKVKUKIwQZW5jb2Rlci4wLndlaWdodJSMDHRvcmNoLl91dGlsc5SMEl9yZWJ1aWxkX3Rl\n"
                    "bnNvcl92MpSTlCiMDXRvcmNoLnN0b3JhZ2WUjBBfbG9hZF9mcm9tX2J5dGVzlJOUQv0DAACAAooK\n"
                    "bPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0\n"
                    "dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRx\n"
                    "BksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpx\n"
                    "AVgPAAAAMTA1NTUzMTMzMzEwNzM2cQJYAwAAAGNwdXEDS8BOdHEEUS6AAl1xAFgPAAAAMTA1NTUz\n"
                    "MTMzMzEwNzM2cQFhLsAAAAAAAAAAs883PebKjj36NOy+zy7IvmXtM78Ve68+XkxDvFQluL6PW7O+\n"
                    "FKigvnXGAr41EGk+Un2CPmy2Kb2uRzE+885KP7kH/L4EcXg+YivkPhKaLD892AI/pzDgvsoFQz+J\n"
                    "iwA/7qsDPwCjEr+GuaC+jQOHPl9EDr72bIc+kNZRvklaAr8RBfU+JZGMvgWfRbtq49M+W8XWvQOq\n"
                    "572EFb4++f8YPwombT78w8U+PLjMvp8Iz742eBI/TU6PvouyBb+zU+O9i2HDvJjpVD/2MM69j1RY\n"
                    "P4kzXD8wjko/O/1RP8NOS7xUaRY+Y5o+P6GSWD85cQU/IwcAvzzJX76+mYK+tIwzvd9cLD9PPaW9\n"
                    "xc8kPqmDDT8W0qU+bQYHvsOspTxiz9o9e+AIvyRxPLvwEIM+x7XLPfCbtL7KCEC/7HvovvQBlD3E\n"
                    "Gxu/V8JNPlNNwb7Xkv++Co2rPQMhVL4xbtk+nZ9qPpBs2j5gn/g+91UgPnC6Jj7/oLe8mBAKP4vU\n"
                    "vr5/QDK/mwEYPwxhAb4BgQi/S50xvtBudz0sFXS+cYiFvjJioz7NtfI8UL06PzJEnb6JOhY9EVO5\n"
                    "vdw7mT4NZEq+QRKFPK7U/r4oniS/bK8bP6Umvj5PDxw/Q/j9va5FbT4QwwE/6RHGPrxERL59cey+\n"
                    "+PMqP6R6oj4wLA8/QMI3vkIb9r7oFcI+K8BXPfgT075j5vw+ma0FP/Dr8r4YqZU80+txvnuXrT7B\n"
                    "vuO+cDxLPelYKz4zfFi9uB+zPL4mn70H4u0+VOywPiyVNb/Tc9A+mGG6PtMhAb+9MXS+7ayWvbzu\n"
                    "jD7nnp0+xy63vUJJ6rvsioq+LlTgPQWmLj9XR48+QVOtPpXwLD/DwP++8rgUPT6V6rw1916+1e0D\n"
                    "v//v5b5kvRC+DmBAPwUD0z47P/W+VCOhvh70pD6R8iU89cnQPk58fr6o0W+9OVhDvAeHx74pT4s+\n"
                    "oIIRv+7ytD1fhAY/aBnAvmb61j6bE+2+rJe1Pqhz5z6BHw6/wSb9PjeRAr8S7dM9lIWUUpRLAEsQ\n"
                    "SwyGlEsMSwGGlIloBClSlHSUUpSMDmVuY29kZXIuMC5iaWFzlGgJKGgMQj0BAACAAooKbPycRvkg\n"
                    "aqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2Vu\n"
                    "ZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQA\n"
                    "AABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAA\n"
                    "MTA1NTUzMTMzMzEwODE2cQJYAwAAAGNwdXEDSxBOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzEw\n"
                    "ODE2cQFhLhAAAAAAAAAADgKqvlaSbz66VCE+Ym88vnP6gz65EQm+4i0DPh5blL47ywS+b/4bPjSs\n"
                    "yT2rMRE+OCjsvRqOOj5z9Gw+XZifvpSFlFKUSwBLEIWUSwGFlIloBClSlHSUUpSMEGVuY29kZXIu\n"
                    "MS53ZWlnaHSUaAkoaAxCAQEAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9j\n"
                    "b2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQo\n"
                    "WAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdl\n"
                    "cQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTA4OTZxAlgDAAAAY3B1cQNL\n"
                    "AU50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTA4OTZxAWEuAQAAAAAAAABUexs/lIWUUpRLAEsB\n"
                    "hZRLAYWUiWgEKVKUdJRSlIwOZW5jb2Rlci4xLmJpYXOUaAkoaAxCAQEAAIACigps/JxG+SBqqFAZ\n"
                    "LoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFu\n"
                    "cQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxv\n"
                    "bmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1\n"
                    "NTMxMzMzMTA5NzZxAlgDAAAAY3B1cQNLAU50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTA5NzZx\n"
                    "AWEuAQAAAAAAAACER/68lIWUUpRLAEsBhZRLAYWUiWgEKVKUdJRSlIwWZW5jb2Rlci4xLnJ1bm5p\n"
                    "bmdfbWVhbpRoCShoDEIBAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2Nv\n"
                    "bF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChY\n"
                    "BQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3JhZ2Vx\n"
                    "AGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMTA1NnECWAMAAABjcHVxA0sB\n"
                    "TnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMTA1NnEBYS4BAAAAAAAAAGy3Kz+UhZRSlEsASwGF\n"
                    "lEsBhZSJaAQpUpR0lFKUjBVlbmNvZGVyLjEucnVubmluZ192YXKUaAkoaAxCAQEAAIACigps/JxG\n"
                    "+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVf\n"
                    "ZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRY\n"
                    "BAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8A\n"
                    "AAAxMDU1NTMxMzMzMTExMzZxAlgDAAAAY3B1cQNLAU50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMz\n"
                    "MTExMzZxAWEuAQAAAAAAAAD07ZtAlIWUUpRLAEsBhZRLAYWUiWgEKVKUdJRSlIwdZW5jb2Rlci4x\n"
                    "Lm51bV9iYXRjaGVzX3RyYWNrZWSUaAkoaAxCBAEAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEA\n"
                    "KFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBl\n"
                    "X3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIo\n"
                    "WAcAAABzdG9yYWdlcQBjdG9yY2gKTG9uZ1N0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMTIxNnEC\n"
                    "WAMAAABjcHVxA0sBTnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMTIxNnEBYS4BAAAAAAAAAKQG\n"
                    "AAAAAAAAlIWUUpRLACkpiWgEKVKUdJRSlIwQZW5jb2Rlci40LndlaWdodJRoCShoDEL9AgAAgAKK\n"
                    "Cmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxp\n"
                    "dHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50\n"
                    "cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UK\n"
                    "cQFYDwAAADEwNTU1MzEzMzMxMTI5NnECWAMAAABjcHVxA0uATnRxBFEugAJdcQBYDwAAADEwNTU1\n"
                    "MzEzMzMxMTI5NnEBYS6AAAAAAAAAADi8or58nqk92K6vPi/XD79foNU+jwGEvt/FFz6vX9o+Y1Ak\n"
                    "P7LUur4rJh2/mY38Pf+QCT9y7fY82SsUvcymAD1Xb56+KmBgvkJuIz9W6xq/EUa4PqyCKD80cSW+\n"
                    "C5DSvQEtkr5s/5E9AgITPy7wKL8Xh/g8UMLjPua18751SY6+SqE2P0X8VL/be+c992AcP67igL+L\n"
                    "gZu9oIhRP3nZUj1sQhE/LxvVPu27/D1teoS+Y2QZP3L4Gj+jWNa8SK5fPSlhJjmHFQc+p8eLP3QI\n"
                    "1D6zltC+c3NhPzzRkT8yQ9K+1fQ5P86kVr+FRvi+BjVOvzIHGj8bFWs/rP8PPkhtED+nFCW/l4BL\n"
                    "PkpcAL9EbM2+CsAePxyOBz+0SQO/UWUYP9eA2b6DYoO+rfkLPzg4Hb9xzye/756WPqU+vz0Z8vu+\n"
                    "M/QAv6AJHz2/Rca8qjitvScC6z57Nrm+5YLRvtCm2D7iqsY99IAvP2L2ET57MRa/Lh0svepup76D\n"
                    "9qk+4Voev5N9PD9c4OG+h1vqPYBVC7+lKpG+IHPOPl7WWz22BYC/QtJFPwmsw74vQjA+6o0JvkJk\n"
                    "pb45Jv6+MP4aPzdqET8NfP4+wFxDvCjlcj712EQ+aqIFv8NyiT644Kg+vcf/PuMjjb6HoK29rKbe\n"
                    "PmizGb+eKKK9yZWUvseARz9oGzM/lIWUUpRLAEsISxCGlEsQSwGGlIloBClSlHSUUpSMDmVuY29k\n"
                    "ZXIuNC5iaWFzlGgJKGgMQh0BAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3Rv\n"
                    "Y29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEE\n"
                    "KFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFn\n"
                    "ZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzExMzc2cQJYAwAAAGNwdXED\n"
                    "SwhOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzExMzc2cQFhLggAAAAAAAAAdIOhvfNb2z1gZYy+\n"
                    "Hdcjv9EtAz0p7kk+e9esvpoV472UhZRSlEsASwiFlEsBhZSJaAQpUpR0lFKUjBBlbmNvZGVyLjUu\n"
                    "d2VpZ2h0lGgJKGgMQgEBAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29s\n"
                    "X3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgF\n"
                    "AAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEA\n"
                    "Y3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzExNDU2cQJYAwAAAGNwdXEDSwFO\n"
                    "dHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzExNDU2cQFhLgEAAAAAAAAAaZUeP5SFlFKUSwBLAYWU\n"
                    "SwGFlIloBClSlHSUUpSMDmVuY29kZXIuNS5iaWFzlGgJKGgMQgEBAACAAooKbPycRvkgaqhQGS6A\n"
                    "Ak3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnEC\n"
                    "iFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25n\n"
                    "cQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUz\n"
                    "MTMzMzExNTM2cQJYAwAAAGNwdXEDSwFOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzExNTM2cQFh\n"
                    "LgEAAAAAAAAAifGuPpSFlFKUSwBLAYWUSwGFlIloBClSlHSUUpSMFmVuY29kZXIuNS5ydW5uaW5n\n"
                    "X21lYW6UaAkoaAxCAQEAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xf\n"
                    "dmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUA\n"
                    "AABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBj\n"
                    "dG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTE2MTZxAlgDAAAAY3B1cQNLAU50\n"
                    "cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTE2MTZxAWEuAQAAAAAAAACqNwW/lIWUUpRLAEsBhZRL\n"
                    "AYWUiWgEKVKUdJRSlIwVZW5jb2Rlci41LnJ1bm5pbmdfdmFylGgJKGgMQgEBAACAAooKbPycRvkg\n"
                    "aqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2Vu\n"
                    "ZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQA\n"
                    "AABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAA\n"
                    "MTA1NTUzMTMzMzExNjk2cQJYAwAAAGNwdXEDSwFOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzEx\n"
                    "Njk2cQFhLgEAAAAAAAAAfReEQJSFlFKUSwBLAYWUSwGFlIloBClSlHSUUpSMHWVuY29kZXIuNS5u\n"
                    "dW1fYmF0Y2hlc190cmFja2VklGgJKGgMQgQBAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChY\n"
                    "EAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9z\n"
                    "aXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgH\n"
                    "AAAAc3RvcmFnZXEAY3RvcmNoCkxvbmdTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTE3NzZxAlgD\n"
                    "AAAAY3B1cQNLAU50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTE3NzZxAWEuAQAAAAAAAACkBgAA\n"
                    "AAAAAJSFlFKUSwApKYloBClSlHSUUpSMEGRlY29kZXIuMC53ZWlnaHSUaAkoaAxC/QIAAIACigps\n"
                    "/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0\n"
                    "bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEG\n"
                    "SwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEB\n"
                    "WA8AAAAxMDU1NTMxMzMzMTE4NTZxAlgDAAAAY3B1cQNLgE50cQRRLoACXXEAWA8AAAAxMDU1NTMx\n"
                    "MzMzMTE4NTZxAWEugAAAAAAAAACl4jq+CmByPs4PCT/D1gO/Bi6zPpr4o76TXn8+/A3Jvg9l+L5J\n"
                    "o/6947EIv9xmq77/wAa/NPh9v85PNr/GSLm+a/0aP81Qgj+o0vq+kuiwPkOIhD+sA2w/JmxmPoCB\n"
                    "A76RXbs+6V1+Pm2FC75r3qm+WcAPvzly2DsC3sC9X8qAvGpQTT67KNY+r23yvRxmyD6BCQm+NG3U\n"
                    "vqeaWT340LE+kg+xvl6HZz4TM7C9UZiGPXeD5L3uybg9ZjiEvnQHaD0NY5c+rupRPZiiDT7TkPo+\n"
                    "Pa2hPnkgFb/6IMs+d10jv83HcD5PSIQ8xQeMPoNFY75Hs8S+slj4vbyGtL6UyYi9cM3zviS6qD5b\n"
                    "Mjy+MhGWPunGDz04EIq+2aNGPlCAfL6j11A+rbafPeVDAr+KaM89erirvg0G/b3hvPO9lM5gvr2P\n"
                    "xz6fbCu9J6UWPZKPeb5obbW+Ab7ePokTWT7essU9tZTXPmwgyD6SfCo/AC0ePgZ//L5/I/Y+VbxI\n"
                    "PgBpMT+r0AC/bCC7PQM6pTxuoxW/TFv3vnISCz+WWQ89FkXlvnKtBT7RSYM+S5oivvK6hL4raNy9\n"
                    "oJ0fPm3HAD9ha7q9B8jovoD1Ob50v5w9v5ZrvsOBbz5lJZY+pmumPiSsUb3Y/7u+sXkSPsjrl77B\n"
                    "0NO8qNWTPVozeb7Kh+m8sUWuPZSFlFKUSwBLEEsIhpRLCEsBhpSJaAQpUpR0lFKUjA5kZWNvZGVy\n"
                    "LjAuYmlhc5RoCShoDEI9AQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2Nv\n"
                    "bF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChY\n"
                    "BQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3JhZ2Vx\n"
                    "AGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMTkzNnECWAMAAABjcHVxA0sQ\n"
                    "TnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMTkzNnEBYS4QAAAAAAAAAA74nr6nhQK+Em9cPlgI\n"
                    "MT5/Q2C+ouOQPjo7yr2uQ8U9QMOCPQMlQT5nqM69QaiQvVepBjsza7u8oukvPW6PgT6UhZRSlEsA\n"
                    "SxCFlEsBhZSJaAQpUpR0lFKUjBBkZWNvZGVyLjEud2VpZ2h0lGgJKGgMQgEBAACAAooKbPycRvkg\n"
                    "aqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2Vu\n"
                    "ZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQA\n"
                    "AABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAA\n"
                    "MTA1NTUzMTMzMzEyMDE2cQJYAwAAAGNwdXEDSwFOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzEy\n"
                    "MDE2cQFhLgEAAAAAAAAACnn8PpSFlFKUSwBLAYWUSwGFlIloBClSlHSUUpSMDmRlY29kZXIuMS5i\n"
                    "aWFzlGgJKGgMQgEBAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3Zl\n"
                    "cnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAA\n"
                    "c2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3Rv\n"
                    "cmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzEyMDk2cQJYAwAAAGNwdXEDSwFOdHEE\n"
                    "US6AAl1xAFgPAAAAMTA1NTUzMTMzMzEyMDk2cQFhLgEAAAAAAAAAQD8FvZSFlFKUSwBLAYWUSwGF\n"
                    "lIloBClSlHSUUpSMFmRlY29kZXIuMS5ydW5uaW5nX21lYW6UaAkoaAxCAQEAAIACigps/JxG+SBq\n"
                    "qFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5k\n"
                    "aWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAA\n"
                    "AGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAx\n"
                    "MDU1NTMxMzMzMTIxNzZxAlgDAAAAY3B1cQNLAU50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTIx\n"
                    "NzZxAWEuAQAAAAAAAADPajw9lIWUUpRLAEsBhZRLAYWUiWgEKVKUdJRSlIwVZGVjb2Rlci4xLnJ1\n"
                    "bm5pbmdfdmFylGgJKGgMQgEBAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3Rv\n"
                    "Y29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEE\n"
                    "KFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFn\n"
                    "ZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzEyMjU2cQJYAwAAAGNwdXED\n"
                    "SwFOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzEyMjU2cQFhLgEAAAAAAAAApVCTP5SFlFKUSwBL\n"
                    "AYWUSwGFlIloBClSlHSUUpSMHWRlY29kZXIuMS5udW1fYmF0Y2hlc190cmFja2VklGgJKGgMQgQB\n"
                    "AACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gN\n"
                    "AAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMA\n"
                    "AABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkxvbmdTdG9y\n"
                    "YWdlCnEBWA8AAAAxMDU1NTMxMzMzMTIzMzZxAlgDAAAAY3B1cQNLAU50cQRRLoACXXEAWA8AAAAx\n"
                    "MDU1NTMxMzMzMTIzMzZxAWEuAQAAAAAAAACkBgAAAAAAAJSFlFKUSwApKYloBClSlHSUUpSMEGRl\n"
                    "Y29kZXIuNC53ZWlnaHSUaAkoaAxC/QMAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAA\n"
                    "cHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVz\n"
                    "cQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABz\n"
                    "dG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTI0MTZxAlgDAAAA\n"
                    "Y3B1cQNLwE50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTI0MTZxAWEuwAAAAAAAAACnzh6+Rt3v\n"
                    "vaL1zz0NDe69RGOTvk/YCD84jYK+LC+jvoBfqL73Jga/sXW0vvmPxz7/7mO+8OMMP6jiQ7t2gNi7\n"
                    "D3z8vt8dhb1T7HU9mi4Av99Xbb4g/wg/Zb0EPbzZLb45v8C+hqC9vsZ7DD+ZVNA+cQwuvnYChr6h\n"
                    "Cbk+WLDAPeyqnb4nnG+9NIFdPSYdEj5YrRu+k43NPmuPn77REni+sibXvoXHkb7TXXC+7RO+PqWN\n"
                    "Qr4/lLC92Pu3PtMpj74AbWC+eWOsvesJmz2XrR8+9Z2xvo2vYj6nPO++nUChvvZs374qYaG+Gn+J\n"
                    "vdTt4T61AdK+ks0uPl/jiT6HGMe8ntwYPZWNR70gmSM9IIjGvmVMKr4P5OA9V+qgvnwknL2GZUq+\n"
                    "HAT7vWP41b5aEaw+avmIvjAd0T7nUYI+HIA3vU7LB78mWRG9hw4hPTiNkr6R7Ti+9mQUP4dODTuQ\n"
                    "Kea9a8Qavvr3C746h5O+Vi6HPoJgCr1se5Y+cczqPuwVzb5YqQ6+Q2+vvUf3nT0/2da+/ZObvoyf\n"
                    "6b5xSQi/MqGQvhRvsb4UIde+SjvlvIRkxj5OFEK+DkGQvM2W/7zsUyE9YRWmvt8qPb0csiw9jW7I\n"
                    "vgNNAL7PGxG91Wz1PNv7O75izJO9Mdu+vi2K4z5h0pc+Z/Cpvibmkj6Iape+GbW8vfRstr5fVme9\n"
                    "xi1IPWaGEr4MuYS+HDh1Pi9Opr517j6++Jpkviero750Se8+t+MtPWFNYz0PWDM+s0M7vlbuHTzA\n"
                    "D22+KK2ovQGojj0hXPy+3maWvv/hgT4tmQC/hVyIvg+Ttr4Oqai+uTYNP103XT7WJr6+4uImP7gN\n"
                    "jb77Q0S9mkINvpZQ5L1xpNI9Q5RHvmwTwL5yLge/DiLCvngItr6XiBK/Qmxxvbjquz13KPE+0acB\n"
                    "v6bvML4hXQw/wB8vPuWEqDxIS4K9TY1lPd6zkr72q0q9tI7DPv37kr6rX0G+kD/pvkS8wL7xKOa+\n"
                    "DYXYPtUFlb5s2Bs/M1q4PI0SKriUhZRSlEsASwxLEIaUSxBLAYaUiWgEKVKUdJRSlIwOZGVjb2Rl\n"
                    "ci40LmJpYXOUaAkoaAxCLQEAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9j\n"
                    "b2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQo\n"
                    "WAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdl\n"
                    "cQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTI0OTZxAlgDAAAAY3B1cQNL\n"
                    "DE50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTI0OTZxAWEuDAAAAAAAAABjHLs+pbsDPy4eDz+C\n"
                    "aMw+QYcmPxKQIz8EU+8+EVQoP/q+Kj8QN8s+3Fu8Pp1OBz+UhZRSlEsASwyFlEsBhZSJaAQpUpR0\n"
                    "lFKUdX2UjAlfbWV0YWRhdGGUaAQpUpQojACUfZSMB3ZlcnNpb26USwFzjAdlbmNvZGVylH2UaNpL\n"
                    "AXOMCWVuY29kZXIuMJR9lGjaSwFzjAllbmNvZGVyLjGUfZRo2ksCc4wJZW5jb2Rlci4ylH2UaNpL\n"
                    "AXOMCWVuY29kZXIuM5R9lGjaSwFzjAllbmNvZGVyLjSUfZRo2ksBc4wJZW5jb2Rlci41lH2UaNpL\n"
                    "AnOMCWVuY29kZXIuNpR9lGjaSwFzjAdkZWNvZGVylH2UaNpLAXOMCWRlY29kZXIuMJR9lGjaSwFz\n"
                    "jAlkZWNvZGVyLjGUfZRo2ksCc4wJZGVjb2Rlci4ylH2UaNpLAXOMCWRlY29kZXIuM5R9lGjaSwFz\n"
                    "jAlkZWNvZGVyLjSUfZRo2ksBc3VzYowUb3B0aW1pemVyX3N0YXRlX2RpY3SUfZQojAVzdGF0ZZR9\n"
                    "lChLAH2UKIwEc3RlcJRNpAaMB2V4cF9hdmeUaAkoaAxC/QMAAIACigps/JxG+SBqqFAZLoACTekD\n"
                    "LoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoA\n"
                    "AAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sE\n"
                    "dXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMz\n"
                    "MTI1NzZxAlgDAAAAY3B1cQNLwE50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTI1NzZxAWEuwAAA\n"
                    "AAAAAABOgCk3CNYrNxleKTfVdyo3uu8pN8mhLzeYMDQ3g3g8N/82PzcAi0A3INU7N/5fNjc35+00\n"
                    "QhnqNEVp3zSXf/c0lHfyNA3T9zTNSQo1+mYNNbnG+jSChe00vVz4NAk3MzWm7Um2RYM1toH47LW3\n"
                    "NrO1izwyMuZMsjXWkmUziJOftZJWLLXiLQ00AWRSNZKnPTVnmBI407gKOLr7Bjj7PQU4k0UKOB+t\n"
                    "Dzgg+BY4gLMfOFT4ITi4wCY4j18rOOo1MjilhXO3eEJzt1T8crcC13K3Mptyt8DGcrfQYHO35th0\n"
                    "t+LOdbeZyna3t4F2tx69drcgFeq44D/ouHp56bhxWeq4Va/ruJVD6rhwLum4bWHnuBjv5rhsxem4\n"
                    "QI/tuN0o8riu8RS3r2oTtykfFLdNRBe3nkUat4J0F7dO8hq3WaEct7KqGrekpxe3h5MXt1K+Gbe1\n"
                    "ZsY4kGy9OLc0tTiedbE4ZCiyOIvxtThRgLg46bW9OJ2UxTjpnc84p27XOB6W5DhD7Q24ElsIuGgs\n"
                    "ArhPWwO4lJUGuMtHB7gUCBG4o4whuF9WKLh9+Cu43YksuPITMLiDYJM4HGiOODHMiTh1i4k4HYGL\n"
                    "ONbujjg735Q4mYubOLm+nzjTpaU47KarONbbszjmI5o3yDaTN3P9gjcN+H43PPyAN3qfhDfbiZg3\n"
                    "nuq3N4UkzDcpGNc32HvTN2Es2jcSJjG3/uF9tjZ5HjMFCgk2YKzCNXZjkzU+mQW2riEOtxn+ZrdN\n"
                    "J5u37Xe+t7EkAbgAWiK3vpEPt/SR+LYK09+2PePItqJyy7beEwe31/03t3G6ULeWk1O3miNNtzMg\n"
                    "XLct2vC4Gb3quB2c5bhbIeC4uAjiuK8f5bixnei4+YPtuII/8Lj9bPS44On4uEjUArn5J+035j3c\n"
                    "N1fbtjdV47E3z3qtNz43uDdLML43zODgNxViATgIGRA4LY4dOONhQjhJ3102WWyNNv69jzZzQYw2\n"
                    "X5KTNpSQ1jZzNOY2G6PvNtkLAjeNCAg30c3xNrpR5zaUhZRSlEsASxBLDIaUSwxLAYaUiWgEKVKU\n"
                    "dJRSlIwKZXhwX2F2Z19zcZRoCShoDEL9AwAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAA\n"
                    "AABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6\n"
                    "ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAA\n"
                    "AHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMjY1NnECWAMA\n"
                    "AABjcHVxA0vATnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMjY1NnEBYS7AAAAAAAAAAH5EbzQz\n"
                    "DG80yKpuNBhHbzS6HXA0tl9xNOiRcTTTznE090xyNFK8cjQcYnI0waByNHg+PDNtqTwzlnY8MxRO\n"
                    "PDPdDTwzq8I7MzebOzPjrTszd387Mz/NOzO2pjszv5s7MzBR3jLWeN0yzPfcMvxT3TL9Od0yIV3d\n"
                    "MvO13DL+Q9wyZBHcMgTr2TKDeNkyepzZMt5ByDVIQcg12wHINSqWxzUQNcc1qObGNQJ1xjVpNcY1\n"
                    "gDDGNbkgxjW91MU1NBPGNRTiNjPbBDczACE3M41HNzMAXzczFmw3M+haNzPXTTcztEM3M6s7NzOn\n"
                    "JzczAi03M+jrvzVupsE1RojCNcmzwjXvs8I1Nl/CNbcEwjWjE8I1FvTBNZEdwTW8gb81xAu+Na9J\n"
                    "ITNPLCEzrdMgM72JIDMtUiAzGREgM+vNHzMboR8z8psfM+B6HzPiTx8z4GcfM4TO9TPrE/ozMMb9\n"
                    "MxlGADS1KwA04l/+M9ok/DNmgPgzGCHzM4UC7jOfQugz+EziM6qYRjQv8Ec0dMBINLtTSDSmC0c0\n"
                    "ZaRFNL+ARTRJcEQ00YlDNJRiQjQyfUE0uwFBNNGbXDa+ulw2ALFcNgRrXDaby1s2/O9aNnvpWTZR\n"
                    "DVk2PIBYNvCVVzarxFY2bEtWNuLGBDS6QgU0FL0FNHi/BTTjvwU0pHwFNDQaBjT/YQU0D70FNFbj\n"
                    "BDQzKwQ0JvoDNMaPMzU8PjY1gP83NejiODX55Tg1ccY3NZyoNTWt9TI1CSAwNSouLTUd6Ck1txsn\n"
                    "NZcA9DP99vMz7OPzM9SZ8zOyxvMzMcjzM9o08zON3/IzJ1zyM5NB8jNrTPIz4xryM/idujUUsrw1\n"
                    "hVu9NUPXvDVGsrs1Tzq6NZ1nuDUpRLY1Hi60NczVsTXeG681eBWtNXG8rTVk5681eBKxNUe8sTUo\n"
                    "1bE1I8iwNTgyrzXZ+qw1taOqNbjWpzVMFqU1w2miNe6cODaOgDg2i4Y4NnOOODbFsjg21ro4NnRv\n"
                    "ODYSSTg2yHs4Nh9VODaNPTg22/43NpSFlFKUSwBLEEsMhpRLDEsBhpSJaAQpUpR0lFKUdUsBfZQo\n"
                    "aPxNpAZo/WgJKGgMQj0BAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29s\n"
                    "X3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgF\n"
                    "AAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEA\n"
                    "Y3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzEyNzM2cQJYAwAAAGNwdXEDSxBO\n"
                    "dHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzEyNzM2cQFhLhAAAAAAAAAA2OEXN+MlMjUk3uy1/xxH\n"
                    "OMVviLeHYwm5Zcc6t3FBFjm6ni644i7fOI1pDTi6FZS4QE2Mt+khJbmiw9E4lSbJNpSFlFKUSwBL\n"
                    "EIWUSwGFlIloBClSlHSUUpRqBgEAAGgJKGgMQj0BAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1x\n"
                    "AChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlw\n"
                    "ZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoAC\n"
                    "KFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzEyODE2\n"
                    "cQJYAwAAAGNwdXEDSxBOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzEyODE2cQFhLhAAAAAAAAAA\n"
                    "X7lSND5UJDPgA8Ay0xOxNWCNIDOBvqE1vPQMMzlmoDPdFyU0bxw9NneC4jObYgY1qkTXM5vkjDWl\n"
                    "un01IloiNpSFlFKUSwBLEIWUSwGFlIloBClSlHSUUpR1SwJ9lCho/E2kBmj9aAkoaAxCAQEAAIAC\n"
                    "igps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABs\n"
                    "aXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGlu\n"
                    "dHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdl\n"
                    "CnEBWA8AAAAxMDU1NTMxMzMzMTI4OTZxAlgDAAAAY3B1cQNLAU50cQRRLoACXXEAWA8AAAAxMDU1\n"
                    "NTMxMzMzMTI4OTZxAWEuAQAAAAAAAADNeR85lIWUUpRLAEsBhZRLAYWUiWgEKVKUdJRSlGoGAQAA\n"
                    "aAkoaAxCAQEAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lv\n"
                    "bnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9y\n"
                    "dHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gK\n"
                    "RmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTI5NzZxAlgDAAAAY3B1cQNLAU50cQRRLoAC\n"
                    "XXEAWA8AAAAxMDU1NTMxMzMzMTI5NzZxAWEuAQAAAAAAAAAE2BM2lIWUUpRLAEsBhZRLAYWUiWgE\n"
                    "KVKUdJRSlHVLA32UKGj8TaQGaP1oCShoDEIBAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAo\n"
                    "WBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVf\n"
                    "c2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihY\n"
                    "BwAAAHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMzA1NnEC\n"
                    "WAMAAABjcHVxA0sBTnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMzA1NnEBYS4BAAAAAAAAALQx\n"
                    "uDmUhZRSlEsASwGFlEsBhZSJaAQpUpR0lFKUagYBAABoCShoDEIBAQAAgAKKCmz8nEb5IGqoUBku\n"
                    "gAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5x\n"
                    "AohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9u\n"
                    "Z3EHSwR1dS6AAihYBwAAAHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1\n"
                    "MzEzMzMxMzEzNnECWAMAAABjcHVxA0sBTnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMzEzNnEB\n"
                    "YS4BAAAAAAAAALzKMDeUhZRSlEsASwGFlEsBhZSJaAQpUpR0lFKUdUsEfZQoaPxNpAZo/WgJKGgM\n"
                    "Qv0CAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3p\n"
                    "A1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsC\n"
                    "WAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0\n"
                    "U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzEzMjE2cQJYAwAAAGNwdXEDS4BOdHEEUS6AAl1xAFgP\n"
                    "AAAAMTA1NTUzMTMzMzEzMjE2cQFhLoAAAAAAAAAA/7EuOGbQFLjbUH83mWgRNtaY5DXzQb62TrL/\n"
                    "OBepVbauw+k3plK8t0oBn7eiBeK10DJQOFz/njbWl922d1ouOHtEqjgNRgu5zId2N95XrjePWDy4\n"
                    "XRbMN7OmADkJJjU36bpbN09H1LWRNCY2OtxHN+tM8Dc34oI3DH7oNSz0BTh+J4w3l6ZWt3MuqjQn\n"
                    "8gU2SVqYt0JOJrRdiHY3b56BtQOWDzaRMZ004VsOtaooHrXqwWY2SZ7xtZShpbUMxXI2GresN97J\n"
                    "vreWl402wD3ONidk1bdniRg2pt+2N2gLGraokss2jBGStrIAsLaew9I1C0LyNjC++jW0uCy2jWoi\n"
                    "N6cXQzm47xy5DI0zOAr5ZThVwkG5AwKLNxpS6ThLy6+3mQpkOH/uabge4Cu43F9tN6FmNDieCJk3\n"
                    "zRmSt/ebqzgqqD+5vu9zOd3qcrfVZ6m4WwmsOKRMjbiaMLC5Y/6PuCb0Tbg+DJo3EqGCNgXlDbhH\n"
                    "gYS4JyWmuNR1mbffCKq4sHQpOLDGXbg5Dlc3H6kSNzeXPTOHlHo2OWXuN+pqUbbOcFc3Do4At6Jc\n"
                    "5ba5m+w18pJaN8XrtDZ0YQq2en9YN5gR5bhFaF45F3octzGun7iDst42ZUNUuDMzh7mjb4+44oB9\n"
                    "uH7fFDgzN0c3pqgMuGCzgrhaAqC4U+OJt24hG7iUhZRSlEsASwhLEIaUSxBLAYaUiWgEKVKUdJRS\n"
                    "lGoGAQAAaAkoaAxC/QIAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xf\n"
                    "dmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUA\n"
                    "AABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBj\n"
                    "dG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTMyOTZxAlgDAAAAY3B1cQNLgE50\n"
                    "cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTMyOTZxAWEugAAAAAAAAACUUhY1V1iVNdEpBjZ51qs1\n"
                    "iD1PNp2aITVnIRM2T2gdNj0MjzUk3SQ17gusM/ctBDX8es81RHwSNXRztjVTZn40J5eONfXjxzWQ\n"
                    "+dc1zh3dNYmWmjbN4ww1piM5Np4jJjaKp8g1CvMtNXSfyDOp8BQ1mLjnNfJrEzUJDRY29h6GNN/h\n"
                    "DTTSDzQ0gF5oNF81gDM8P2k02+/vMxJYaDTGrIk0mVmbM64E5TPDXdUxxMgYMxyqRDS2mqszMnEM\n"
                    "NATMsDJ4kQw0p6EnNBaaujN8Jv8yb7FvNI/7JDMTgCY06RfiM4VR7zLqrRkzDrSJMaGc+TE+9FEz\n"
                    "njYXMxmXOzNee4MyI/6zNU2S9DUHSKw1R9l1NQgEfTag46E1zrInNiqFpjUp+W81Gf1MNXO/wjPs\n"
                    "Peg0h+SVNc5FYjU8MqY1XAjLNI4t7Ta+2WY3tIKXNwsATzeQ+AA4LcLXNvjxtTfP+pc3pFFXN4oj\n"
                    "zjZ0W401VET0NipFhTcDfsY2g6igN+nZMTZRQRA0taFzNO8ivDQfWXY03FkKNZv/AzQbSws1Wszt\n"
                    "My/lmzPNOekzUyRcMjLCrDOvZCk0Qf5YNL5KFjS72DAzoPmwNTbJ9zXxMvk0O4lZNNHdJDZmZcA0\n"
                    "XsMFNoFWYjWybpY0KfKqNCtwFDQ8GTEzbfMTNYWIojS5rxA02VdHNJSFlFKUSwBLCEsQhpRLEEsB\n"
                    "hpSJaAQpUpR0lFKUdUsFfZQoaPxNpAZo/WgJKGgMQh0BAACAAooKbPycRvkgaqhQGS6AAk3pAy6A\n"
                    "An1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAA\n"
                    "dHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1\n"
                    "LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzEz\n"
                    "Mzc2cQJYAwAAAGNwdXEDSwhOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzEzMzc2cQFhLggAAAAA\n"
                    "AAAALU4fuP5BprivKau2OPr1t5cFi7lsz3M5fzdXuLwWdzmUhZRSlEsASwiFlEsBhZSJaAQpUpR0\n"
                    "lFKUagYBAABoCShoDEIdAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2Nv\n"
                    "bF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChY\n"
                    "BQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3JhZ2Vx\n"
                    "AGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMzQ1NnECWAMAAABjcHVxA0sI\n"
                    "TnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMzQ1NnEBYS4IAAAAAAAAAH9hFTbDwlk2fp/ANJPM\n"
                    "hzSCn2U2snDwN1vLFDUDczA2lIWUUpRLAEsIhZRLAYWUiWgEKVKUdJRSlHVLBn2UKGj8TaQGaP1o\n"
                    "CShoDEIBAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9u\n"
                    "cQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0\n"
                    "cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3JhZ2VxAGN0b3JjaApG\n"
                    "bG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMzUzNnECWAMAAABjcHVxA0sBTnRxBFEugAJd\n"
                    "cQBYDwAAADEwNTU1MzEzMzMxMzUzNnEBYS4BAAAAAAAAAFQ3hreUhZRSlEsASwGFlEsBhZSJaAQp\n"
                    "UpR0lFKUagYBAABoCShoDEIBAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90\n"
                    "b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31x\n"
                    "BChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3Jh\n"
                    "Z2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMzYxNnECWAMAAABjcHVx\n"
                    "A0sBTnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMzYxNnEBYS4BAAAAAAAAAD8YBTeUhZRSlEsA\n"
                    "SwGFlEsBhZSJaAQpUpR0lFKUdUsHfZQoaPxNpAZo/WgJKGgMQgEBAACAAooKbPycRvkgaqhQGS6A\n"
                    "Ak3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnEC\n"
                    "iFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25n\n"
                    "cQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUz\n"
                    "MTMzMzEzNjk2cQJYAwAAAGNwdXEDSwFOdHEEUS6AAl1xAFgPAAAAMTA1NTUzMTMzMzEzNjk2cQFh\n"
                    "LgEAAAAAAAAAticgOpSFlFKUSwBLAYWUSwGFlIloBClSlHSUUpRqBgEAAGgJKGgMQgEBAACAAooK\n"
                    "bPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gNAAAAbGl0\n"
                    "dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRx\n"
                    "BksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpx\n"
                    "AVgPAAAAMTA1NTUzMTMzMzEzNzc2cQJYAwAAAGNwdXEDSwFOdHEEUS6AAl1xAFgPAAAAMTA1NTUz\n"
                    "MTMzMzEzNzc2cQFhLgEAAAAAAAAAxjYjOJSFlFKUSwBLAYWUSwGFlIloBClSlHSUUpR1Swh9lCho\n"
                    "/E2kBmj9aAkoaAxC/QIAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xf\n"
                    "dmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUA\n"
                    "AABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBj\n"
                    "dG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTM4NTZxAlgDAAAAY3B1cQNLgE50\n"
                    "cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTM4NTZxAWEugAAAAAAAAADHJB+5xjZMudbeRzUxjt41\n"
                    "XDmhucXinrkqVyW3/ILytxxgjDmpL8U5h+UZtsAiQLV+cR46Vw0LOvULuTfR+Is4oDccuWEFWrm6\n"
                    "CgY2J6BoNRAYsLmS+Zm5TJAet00U9LdN7Q+5XVlQuWZJuTYIWeA29YC4uT0anrlnIic2ZJaCOH4Y\n"
                    "Xrm/NKG5EvWdNYBGnzbdJQa6zKzuuXJfljZiTBU3RRWNOaQm9TmKXJS26uectgtEOTrP5iw6ZGrA\n"
                    "Nx5NWji7kIq5ZH3OuSGbqLbeGhI29colum30F7qX/BC4KbXYuJYWRLlvApC5z8N2s3bzGTZEe+65\n"
                    "VqnRuS7UILcX4GG3RMOauRQ9zbkXWwC2O1XvNflYKLrI9Rq6+ZYPt+a1MrjEr1q5L1mcubAWw7Zb\n"
                    "isw2/FQFukOO87mqX7y3XNfPt/giazntuFk5e1teNfSRVzYfE5o5T+maOWrgszcrQSU4SgEmOkO9\n"
                    "bDpmzlQ2SzPKtqiPuTrVbaY6gTyfOL1+EjnW0IO5iLevuTxiLrYXSAU2EEwOuudG+bmYZRK49R5m\n"
                    "uAfTpjmTp/c5klAAt3NX/7ZvjTs6PXElOncQ0zeidJa3PZPlOD8MXzlA8121MIC3trThyDnrILU5\n"
                    "7x+JNxN2Gzg7KnI4iV6jOAPNgjPxjmWzTjgPOREe4DiwCZ42M97+N5SFlFKUSwBLEEsIhpRLCEsB\n"
                    "hpSJaAQpUpR0lFKUagYBAABoCShoDEL9AgAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAA\n"
                    "AABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6\n"
                    "ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAA\n"
                    "AHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxMzkzNnECWAMA\n"
                    "AABjcHVxA0uATnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxMzkzNnEBYS6AAAAAAAAAAFxdzzX/\n"
                    "Gzg2qAb7MGGDry+t+x43nCT1NhGR9jGCjfYzr1dFNi30XjaebcYwa5a2L2BULDcdehs3SOomMru4\n"
                    "xzP2b9c2YT4QNp3d8jFhqIwwjOUeN93xlDcYTx0xAYiBNOD5ZTe7gX024eMZMjqBgTGtU2c33Q8b\n"
                    "OP96UDIriIk1e8v3N9dT5zYJ2kgyfGtOMWllXje7neY3pVhJMmCOnzWUuM43miEdN8OdTDJToSwx\n"
                    "qwgHOOBWOzho9pkyZGV/NTI1mDcK/eQ2Ahx9MnvvDjFBvZc3v1eqN+LwqzJ9BMM0KtXSNTbN2TUu\n"
                    "jxwx3ZbZL9aPuzb1XKk2Mo34MdWf2zPMhqU3C/sqN50rYjK7b0AxYaD7NzMiJTgsxQEz+C00NUzL\n"
                    "LzdUycM22cKHMZqcUTAwioI3lOR5Ny5lgDKxsNg0KI2uNYRdxTVXDicwREZkL3emnjYOvJI2h+Gf\n"
                    "McAYnTN3kcI3uaCbN3a8kjIkBpIxqNODOD+Pkjiy/1kzifaNNfdHIjZXqw42coEtMipeujDsRuk2\n"
                    "kxBDN624LzLOpbI0VnVON1foCDe+yLoyx0JCMROf3TfHRSE4wlTTMiQRwjUItMw2ZPCBNkVA4zDk\n"
                    "XyowOaIxNxfIJTc25i4yS0SGNPUY2zXt+1o11z3vMSfOLDAhjnw2iCn2Nv93ATEhSfQzlIWUUpRL\n"
                    "AEsQSwiGlEsISwGGlIloBClSlHSUUpR1Swl9lCho/E2kBmj9aAkoaAxCPQEAAIACigps/JxG+SBq\n"
                    "qFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5k\n"
                    "aWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAA\n"
                    "AGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAx\n"
                    "MDU1NTMxMzMzMTQwMTZxAlgDAAAAY3B1cQNLEE50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTQw\n"
                    "MTZxAWEuEAAAAAAAAABIZZ65uK8DOtTtjrmBR2u5WMzNucoRHDpX9xu6RjO8udJsDboQoty59iCe\n"
                    "OdqrnzqX3PW5cncMOgtEoznD3OY4lIWUUpRLAEsQhZRLAYWUiWgEKVKUdJRSlGoGAQAAaAkoaAxC\n"
                    "PQEAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekD\n"
                    "WA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJY\n"
                    "AwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRT\n"
                    "dG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTQwOTZxAlgDAAAAY3B1cQNLEE50cQRRLoACXXEAWA8A\n"
                    "AAAxMDU1NTMxMzMzMTQwOTZxAWEuEAAAAAAAAABitJY2Sa/yNr3wGTeAmrY37N3CN07W9zclFp83\n"
                    "DiFiNmwD/jckCEY3lPooNnpuVDiwK+c2f+XKN0LH7zYcjKc2lIWUUpRLAEsQhZRLAYWUiWgEKVKU\n"
                    "dJRSlHVLCn2UKGj8TaQGaP1oCShoDEIBAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAA\n"
                    "AABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6\n"
                    "ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAA\n"
                    "AHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxNDE3NnECWAMA\n"
                    "AABjcHVxA0sBTnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxNDE3NnEBYS4BAAAAAAAAALZavzuU\n"
                    "hZRSlEsASwGFlEsBhZSJaAQpUpR0lFKUagYBAABoCShoDEIBAQAAgAKKCmz8nEb5IGqoUBkugAJN\n"
                    "6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohY\n"
                    "CgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EH\n"
                    "SwR1dS6AAihYBwAAAHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEz\n"
                    "MzMxNDI1NnECWAMAAABjcHVxA0sBTnRxBFEugAJdcQBYDwAAADEwNTU1MzEzMzMxNDI1NnEBYS4B\n"
                    "AAAAAAAAAB5IiTqUhZRSlEsASwGFlEsBhZSJaAQpUpR0lFKUdUsLfZQoaPxNpAZo/WgJKGgMQgEB\n"
                    "AACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3pA1gN\n"
                    "AAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMA\n"
                    "AABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3Rv\n"
                    "cmFnZQpxAVgPAAAAMTA1NTUzMTMzMzE0MzM2cQJYAwAAAGNwdXEDSwFOdHEEUS6AAl1xAFgPAAAA\n"
                    "MTA1NTUzMTMzMzE0MzM2cQFhLgEAAAAAAAAAxgkDvJSFlFKUSwBLAYWUSwGFlIloBClSlHSUUpRq\n"
                    "BgEAAGgJKGgMQgEBAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3Zl\n"
                    "cnNpb25xAU3pA1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAA\n"
                    "c2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3Rv\n"
                    "cmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzE0NDE2cQJYAwAAAGNwdXEDSwFOdHEE\n"
                    "US6AAl1xAFgPAAAAMTA1NTUzMTMzMzE0NDE2cQFhLgEAAAAAAAAAlOHHOpSFlFKUSwBLAYWUSwGF\n"
                    "lIloBClSlHSUUpR1Swx9lCho/E2kBmj9aAkoaAxC/QMAAIACigps/JxG+SBqqFAZLoACTekDLoAC\n"
                    "fXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0\n"
                    "eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUu\n"
                    "gAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTQ0\n"
                    "OTZxAlgDAAAAY3B1cQNLwE50cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTQ0OTZxAWEuwAAAAAAA\n"
                    "AABeB0g5pRghuoPFVDpB/RQ5oDYFuMQRJLjnyL84PofLuOM2nDhWR4O4jDKNt6Y+47cirvo4SqII\n"
                    "uDqAz7ho2884YZcUuf3orrkd8gk67aiTuJkh/Lg6rxi2907wOHyRIbi6+bG4w0DTuM6mnDb8RRA5\n"
                    "xj2/N/ClBLmFfLO2+jGLOHrEm7hxSq+5dGQdOuIDHjly7hS4N2bdta5M7LhraTe5W08WuYe4e7iN\n"
                    "Za41iBkdOY+kxLe/pra4gKEjt76W+DfiCSo45orzuUiuPTq1W1k5V8BHueJUJ7hj3Am5mtBDuanU\n"
                    "wrgQ4Vw2FU5dNn/RETnrEhC55d+QuDmZY7ie75k4mr/3OKDV5LmrMP45pCW1uBQxFrkyoGe38CTw\n"
                    "uCmvjreyMxa4757xtjD6Wbc3bgE5HQMLubYy7Df8QqI3sbwJODQBV7l9cxm5JLf+OWWBRrjFXDe5\n"
                    "PYa2NxYymzhzo564rigjtz012rf8O9O2vxGVOBdTmDg+MnU3MK84ONikCLfPsMs4gQcNupyZPjrq\n"
                    "ure3gdcvuWpxubg77ym559AcuWl4TLgd2Ke4vqANt10CojjoGDM4MirquJZQoLinSpk4Eg7quI3X\n"
                    "x7lngBU6/CRPuGD1d7hjoCW4xPbjOPQEFbmdBaU3UhMQuWWOyzeKnhk5B3NXuX0gqDcsJsq4fCnF\n"
                    "NgQG77il7965GG4UOgEPGTiO/GG5cq9Qt2Zj7Lj40AW5FkpSuAAMv7hLOCk31KOHuCcZMDmxFr23\n"
                    "qYekuAY8BThFACc4ffAZuga3NTpCbzC4xXlGuX13JLjUbgy5tPwiub8Rh7jZos23br4oN0Q/lbch\n"
                    "ygq5VTAAOAy6+bi/dTo4S1xAOaERBrrzEWU6NLLkOJnRCLn9rAK5Lx2qNo/mE7lJBc+4Ge9nOeJO\n"
                    "arf+sls4n40duWVXO7kIyz+4QljwOH/IIDnVYs+5T18kOrmsArfTic04KvBJt4ivWrhGbtG4BBEP\n"
                    "udPd4bgFGJC3Y5wVOUsF47gJb+43Su5AuNc9bziUhZRSlEsASwxLEIaUSxBLAYaUiWgEKVKUdJRS\n"
                    "lGoGAQAAaAkoaAxC/QMAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xf\n"
                    "dmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUA\n"
                    "AABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBj\n"
                    "dG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxMzMzMTQ1NzZxAlgDAAAAY3B1cQNLwE50\n"
                    "cQRRLoACXXEAWA8AAAAxMDU1NTMxMzMzMTQ1NzZxAWEuwAAAAAAAAACEXRo3rDHQN86oADgoyr82\n"
                    "zCeFNjvwYzbnTDg3I0SaNQXyNTY2gGM121IWN7/ZsjdwSCs2MKSpNhZtcjbOkQg45AiwNcfcNjdb\n"
                    "KW83i3hrNs0DmDV8xOs1HotHNrOMnjV6Ju00E9+yNJQ5sDXqG8E2/BIENrelKTZd0y01NdCDN8TA\n"
                    "6jWSqtk21qpEN0BKDDahxtM1ZFuFNGBlwTVIx2s1MvdONQdsyTSCrqc1F1ypNrOnXTWjjRc1qmGW\n"
                    "NHJTRDbOgOc1HWwkN07hbDciVDM2YW6SNTqjpzUUyaA1Hdh5NcFw4TQ1j9Q0qdDdNWaSujbkSaY1\n"
                    "82mUNYRlTzUZCk83M3p4NmN/Tjd5tog3WNieNoScjjVpQt81X7EBNgfPrTV+g1g105SXNPCYmzZ+\n"
                    "7fM2a/u2NYFpjDWCOFI1kQAlN8q/2TX3D7g3zn5fN8OpajZ1Po81JWYSNvzALDb2b5I1ueR4NdID\n"
                    "uzSwt1g20b/ANmiTKDbnOKA1vZJaNcHRATd9U1k2SS4vNzpWcDfHY6A2AluaNS8FxjVpSvk1/K2K\n"
                    "NR/VQjUKPJU05YdmNkdO5Dawiew11379NTTJVTUaE4s3X7pRNe5P3jbG8yM35IBCNSasATZ9ywQ1\n"
                    "jhpCNuI1PjWI1S41IvCJNN9KFDU82VM2AafnNUxf2TRkDfc1PyJ6NiU+VjWIxzw3TSB/N4ZgHDYZ\n"
                    "SZw1K13ONSdrTjXGIYg1kPL4NJ+m4zRzxmA1/XMJN8dmTTYX79Y1+ziiNTuETTeoHOw1EE1QN9SJ\n"
                    "sTcAdkM2nMeRNVPJjzUh3rM1VpBfNcex0TTCY8s0DUSANfQctTbgMYc1hDM5Nei0qjWpgSo3fc7q\n"
                    "NmapwTdc7dg3KrKvNkZnPDbJASI2lcYGN/nEmzWHdqg1IiKaNSjvxjZJ8YU3bGLaNYrFtDYTWBU2\n"
                    "CPscOB6eejb76gs349tDN30/Ija2WiA2Ui9XNfOrGDb4Row1/pY6NSxdjzSF5X82pGemNr8pMjXL\n"
                    "TwE1v8kvNQBBIjeUhZRSlEsASwxLEIaUSxBLAYaUiWgEKVKUdJRSlHVLDX2UKGj8TaQGaP1oCSho\n"
                    "DEItAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN\n"
                    "6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVL\n"
                    "AlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3JhZ2VxAGN0b3JjaApGbG9h\n"
                    "dFN0b3JhZ2UKcQFYDwAAADEwNTU1MzEzMzMxNDY1NnECWAMAAABjcHVxA0sMTnRxBFEugAJdcQBY\n"
                    "DwAAADEwNTU1MzEzMzMxNDY1NnEBYS4MAAAAAAAAADUN6Lq0J3O65ZZFulqenbq8tAi6jTAAukfz\n"
                    "mbo7+vq5qYIkunq5i7qGT+S6LbZgupSFlFKUSwBLDIWUSwGFlIloBClSlHSUUpRqBgEAAGgJKGgM\n"
                    "Qi0BAACAAooKbPycRvkgaqhQGS6AAk3pAy6AAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3p\n"
                    "A1gNAAAAbGl0dGxlX2VuZGlhbnECiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRxBUsC\n"
                    "WAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0\n"
                    "U3RvcmFnZQpxAVgPAAAAMTA1NTUzMTMzMzE0NzM2cQJYAwAAAGNwdXEDSwxOdHEEUS6AAl1xAFgP\n"
                    "AAAAMTA1NTUzMTMzMzE0NzM2cQFhLgwAAAAAAAAA9GhFOMPQqDcw/lA3G1yxN5uVlzeUOZA3vwfK\n"
                    "N3dPQjck9pE3jGGqN6mnMjgfwY43lIWUUpRLAEsMhZRLAYWUiWgEKVKUdJRSlHV1jAxwYXJhbV9n\n"
                    "cm91cHOUXZR9lCiMAmxylEc/UGJN0vGp/IwFYmV0YXOURz/szMzMzMzNRz/v987ZFocrhpSMA2Vw\n"
                    "c5RHPkV5juIwjDqMDHdlaWdodF9kZWNheZRLAIwHYW1zZ3JhZJSJjAhtYXhpbWl6ZZSJjAZwYXJh\n"
                    "bXOUXZQoSwBLAUsCSwNLBEsFSwZLB0sISwlLCksLSwxLDWV1YXWMCnRocmVzaG9sZHOUjBVudW1w\n"
                    "eS5jb3JlLm11bHRpYXJyYXmUjAxfcmVjb25zdHJ1Y3SUk5SMBW51bXB5lIwHbmRhcnJheZSTlEsA\n"
                    "hZRDAWKUh5RSlChLAUsBhZRq/AEAAIwFZHR5cGWUk5SMAmY4lImIh5RSlChLA4wBPJROTk5K////\n"
                    "/0r/////SwB0lGKJQwjp5y3oX8ncP5R0lGKMCWVycl9zdGF0c5R9lCiMBG1lYW6UavsBAABq/gEA\n"
                    "AEsAhZRqAAIAAIeUUpQoSwFLAYWUaggCAACJQwiTKv0JGrWMP5R0lGKMA3N0ZJRq+wEAAGr+AQAA\n"
                    "SwCFlGoAAgAAh5RSlChLAUsBhZRqCAIAAIlDCLipPmXPl8I/lHSUYnV1Lg==\n",
                ),
                mlflow.entities.Param("model_key", "sandbox:lolsadsd::sad:lolsadasdsda"),
                mlflow.entities.Param(
                    "secondary_artifacts",
                    "gASVUQIAAAAAAAB9lIwGc2NhbGVylIwQc2tsZWFybi5waXBlbGluZZSMCFBpcGVsaW5llJOUKYGU\n"
                    "fZQojAVzdGVwc5RdlIwOc3RhbmRhcmRzY2FsZXKUjBtza2xlYXJuLnByZXByb2Nlc3NpbmcuX2Rh\n"
                    "dGGUjA5TdGFuZGFyZFNjYWxlcpSTlCmBlH2UKIwJd2l0aF9tZWFulIiMCHdpdGhfc3RklIiMBGNv\n"
                    "cHmUiIwObl9mZWF0dXJlc19pbl+USwKMD25fc2FtcGxlc19zZWVuX5SMFW51bXB5LmNvcmUubXVs\n"
                    "dGlhcnJheZSMBnNjYWxhcpSTlIwFbnVtcHmUjAVkdHlwZZSTlIwCaTiUiYiHlFKUKEsDjAE8lE5O\n"
                    "Tkr/////Sv////9LAHSUYkMIBAAAAAAAAACUhpRSlIwFbWVhbl+UaBSMDF9yZWNvbnN0cnVjdJST\n"
                    "lGgXjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwKFlGgZjAJmOJSJiIeUUpQoSwNoHU5OTkr/\n"
                    "////Sv////9LAHSUYolDEAAAAAAAAOA/AAAAAAAA4D+UdJRijAR2YXJflGgkaCZLAIWUaCiHlFKU\n"
                    "KEsBSwKFlGguiUMQAAAAAAAA0D8AAAAAAADQP5R0lGKMBnNjYWxlX5RoJGgmSwCFlGgoh5RSlChL\n"
                    "AUsChZRoLolDEAAAAAAAAOA/AAAAAAAA4D+UdJRijBBfc2tsZWFybl92ZXJzaW9ulIwFMS4xLjGU\n"
                    "dWKGlGGMBm1lbW9yeZROjAd2ZXJib3NllIloQGhBdWJzLg==\n",
                ),
            ],
        ),
    )
