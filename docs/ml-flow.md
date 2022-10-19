# MLflow

Numalogic has built in support for Mlflow's tracking and logging system.

### Starting MLflow

To start the [mlflow server on localhost](https://www.mlflow.org/docs/latest/tracking.html#scenario-1-mlflow-on-localhost),
which has already been installed optionally via `poetry` dependency, run the following command.

Replace the `{directory}` with the path you want to save the models.

```shell
mlflow server \
        --default-artifact-root {directory}/mlruns --serve-artifacts \
        --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Once the mlflow server has been started, you can navigate to http://127.0.0.1:5000/ to explore mlflow UI.

Numalogic provides `MLflowRegistrar`, to save and load models to/from MLflow.

### Model saving

Here, `tracking_uri` is the uri where mlflow server is running. The `static_keys` and `dynamic_keys` are used to form a unique key for the model.

The `primary_artifact` would be the main model, and `secondary_artifacts` can be used to save any pre-processing models like scalers. 

```python
from numalogic.registry import MLflowRegistrar

# static and dynamic keys are used to look up a model
static_keys = ["synthetic", "3ts"]
dynamic_keys = ["minmaxscaler", "sparseconv1d"]

registry = MLflowRegistrar(tracking_uri="http://0.0.0.0:5000", artifact_type="pytorch")
registry.save(
   skeys=static_keys, 
   dkeys=dynamic_keys, 
   primary_artifact=model, 
   secondary_artifacts={"preproc": scaler}
)
```

### Model loading

Once, the models are save to MLflow, the `load` function of `MLflowRegistrar` can be used to load the model.

```python
registry = MLflowRegistrar(tracking_uri="http://0.0.0.0:8080")
artifact_dict = registry.load(
    skeys=static_keys, dkeys=dynamic_keys
)
scaler = artifact_dict["secondary_artifacts"]["preproc"]
model = artifact_dict["primary_artifact"]
```

For more details, please refer to [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html#)