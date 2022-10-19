# Quick Start

## Installation

Install Numalogic and experiment with different tools available.

```shell
pip install numalogic
```

## Numalogic as a Library

Numalogic can used as an independent library. 

```python
from numalogic.models.autoencoder import SparseAEPipeline
from numalogic.models.autoencoder.variants import Conv1dAE

X_train = []

#
model = SparseAEPipeline(
    model=Conv1dAE(in_channels=3, enc_channels=8), seq_len=36, num_epochs=30
)
model.fit(X_train)

#
X_test = []

#
recon = model.predict(X_test)
anomaly_score = model.score(X_test)

```

For more detailed experimentation, refer to [quick-start-example](https://github.com/numaproj/numalogic/blob/main/examples/quick-start.ipynb)

## Numalogic as streaming ML using Numaflow

### Running Numaflow:

 ```
   kubectl create ns numaflow-system
   kubectl apply -n numaflow-system -f https://raw.githubusercontent.com/numaproj/numaflow/stable/config/install.yaml
   kubectl apply -f https://raw.githubusercontent.com/numaproj/numaflow/stable/examples/0-isbsvc-jetstream.yaml
   ```
For more information, refer to https://numaflow.numaproj.io/quick-start/

### Running the Simple Numalogic Pipeline

1. Build the docker image, and push
```
docker build -t simple-numalogic-pipeline . && k3d image import docker.io/library/simple-numalogic-pipeline
```
2. Apply the pipeline
```
kubectl apply -f simple-numalogic-pipeline.yaml
```

### Sending data to the pipeline for ML Inference

1. Port-forward to the http-source vertex
   ```
   kubectl port-forward simple-numalogic-pipeline-in-0-xxxxx 8443
   ```
   
2. Send the data to the pod via curl
   ```
   curl -kq -X POST https://localhost:8443/vertices/in -d '{"data":[0.9,0.1,0.2,0.9,0.9,0.9,0.9,0.8,1,0.9,0.9,0.7]}'
   ```
   
3. Check the anomaly score in the output log 
   ```
   kubectl logs -f simple-numalogic-pipeline-out-0-xxxxx
   ```

### To see the model in ML Flow UI

1. Port forward mlflow-service
   ```
   kubectl port-forward svc/mlflow-service 5000
   ```
2. Navigate to http://127.0.0.1:5000/

### Train on your own data
If you want to train ML model on your own data, replace the `train_data.csv` file with your own file.
The `train_data.csv` file is present under directory `ml_steps/resources` 



