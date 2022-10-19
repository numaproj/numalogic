# Simplepl

## Running Example: Simple Numalogic Pipeline

### Running Numaflow
1. ```
   kubectl create ns numaflow-system
   kubectl apply -n numaflow-system -f https://raw.githubusercontent.com/numaproj/numaflow/stable/config/install.yaml
   kubectl apply -f https://raw.githubusercontent.com/numaproj/numaflow/stable/examples/0-isbsvc-jetstream.yaml
   ```
For more information, refer to https://numaflow.numaproj.io/quick-start/

###Running the Simple Numalogic Pipeline
1. Build the docker image, and push
```
docker build -t simple-numalogic-pipeline:v1 . && k3d image import docker.io/library/simple-numalogic-pipeline:v1
```
2. Apply the pipeline
```
kubectl apply -f simple-numalogic-pipeline.yaml
```

## How to send the data to pipeline?

1. Port-forward to the http-source vertex.
   ```
   kubectl port-forward simple-numalogic-pipeline-in-0-xxxxx 8443
   ```
   
2. Send the data to the pod via curl:
   ```
   curl -kq -X POST https://localhost:8443/vertices/in -d '{"data":[0.9,0.1,0.2,0.9,0.9,0.9,0.9,0.8,1,0.9,0.9,0.7]}'
   ```
3. Check the anomaly score in the output log 
   ```
   kubectl logs -f simple-numalogic-pipeline-out-0-xxxxx
   ```
   
## To edit the udf
### Build locally

1. Install [Poetry](https://python-poetry.org/docs/):
    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```
2. To activate virtual env:
    ```
    poetry shell
    ```
3. To install dependencies:
    ```
    poetry install
    ```


## To see the model in mlflow UI 

1. Port forward mlflow-service
   ```
   kubectl port-forward svc/mlflow-service 5000
   ```
2. Open http://127.0.0.1:5000/

## Train on your own data
If you want to train ML model on your own data, replace the `train_data.csv` file with your own file.
The `train_data.csv` file is present under directory `ml_steps/resources` 