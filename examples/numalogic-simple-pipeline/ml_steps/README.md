# Simple Numalogic Pipeline 


## Installation

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

## Running Example: Simple Numalogic Pipeline
1. Build the docker image, and push
```
make image
# Privilege requried
docker push quay.io/numaio/simplesink-example:python
```
2. Apply the pipeline
```
kubectl apply -f simple-numalogic-pipeline.yaml
```
