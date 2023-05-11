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

### Model saving

Numalogic provides `MLflowRegistry`, to save and load models to/from MLflow.

Here, `tracking_uri` is the uri where mlflow server is running. The `static_keys` and `dynamic_keys` are used to form a unique key for the model.

The `artifact` would be the model or transformer object that needs to be saved.
A dictionary of metadata can also be saved along with the artifact.
```python
from numalogic.registry import MLflowRegistry
from numalogic.models.autoencoder.variants import VanillaAE

model = VanillaAE(seq_len=10)

# static and dynamic keys are used to look up a model
static_keys = ["model", "autoencoder"]
dynamic_keys = ["vanilla", "seq10"]

registry = MLflowRegistry(tracking_uri="http://0.0.0.0:5000")
registry.save(
    skeys=static_keys, dkeys=dynamic_keys, artifact=model, seq_len=10, lr=0.001
)
```

### Model loading

Once, the models are save to MLflow, the `load` function of `MLflowRegistry` can be used to load the model.

```python
from numalogic.registry import MLflowRegistry

static_keys = ["model", "autoencoder"]
dynamic_keys = ["vanilla", "seq10"]

registry = MLflowRegistry(tracking_uri="http://0.0.0.0:8080")
artifact_data = registry.load(
    skeys=static_keys, dkeys=dynamic_keys, artifact_type="pytorch"
)

# get the model and metadata
model = artifact_data.artifact
model_metadata = artifact_data.metadata
```

For more details, please refer to [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html#)
