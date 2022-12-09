# Quick Start

## Installation

Install Numalogic and experiment with the different tools available.

```shell
pip install numalogic
```

## Numalogic as a Library

Numalogic can be used as an independent library, and it provides various ML models and tools. Here, we are using a `AutoencoderPipeline`. Refer to [training section](autoencoders.md) for other available options. 

In this example, the train data set has numbers ranging from 1-10. Whereas in the test data set, there are data points that go out of this range, which the algorithm should be able to detect as anomalies.

```python
import numpy as np
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.models.threshold._std import StdDevThreshold
from numalogic.postprocess import tanh_norm
from numalogic.preprocess.transformer import LogTransformer

X_train = np.array([1, 3, 5, 2, 5, 1, 4, 5, 1, 4, 5, 8, 9, 1, 2, 4, 5, 1, 3]).reshape(-1, 1)
X_test = np.array([-20, 3, 5, 40, 5, 10, 4, 5, 100]).reshape(-1, 1)

# preprocess step
clf = LogTransformer()
train_data = clf.fit_transform(X_train)
test_data = clf.transform(X_test)

# Define threshold estimator and call fit()
thresh_clf = StdDevThreshold(std_factor=1.2)
thresh_clf.fit(train_data)

ae_pl = AutoencoderPipeline(
    model=Conv1dAE(in_channels=1, enc_channels=4), seq_len=8, num_epochs=30
)
# fit method trains the model on train data set
ae_pl.fit(X_train)

# predict method returns the reconstruction error
anomaly_score = ae_pl.score(X_test)

# recalibrate score based on threshold estimator
anomaly_score = thresh_clf.predict(anomaly_score)

# normalizing scores to range between 0-10
anomaly_score_norm = tanh_norm(anomaly_score)
print("Anomaly Scores:", anomaly_score_norm)
```

Below is the sample output, which has logs and anomaly scores printed. Notice the anomaly score for points -20, 40 and 100 in `X_test` is high.
```shell
...snip training logs...
Anomaly Scores: [[6.4051428 ]
 [5.56049277]
 [6.17384938]
 [9.3043446 ]
 [0.22345986]
 [0.48584632]
 [3.18197182]
 [6.29744181]
 [9.99937961]]
```

Replace `X_train` and `X_test` with your own data, and see the anomaly scores generated.

For more detailed experimentation, refer to [quick-start example](https://github.com/numaproj/numalogic/blob/main/examples/quick-start.ipynb)

## Numalogic as streaming ML using Numaflow

Numalogic can also be paired with our streaming platform [Numaflow](https://numaflow.numaproj.io/), to build streaming ML pipelines where Numalogic can be used in [UDF](https://numaflow.numaproj.io/user-defined-functions/).

### Prerequisite

- [Numaflow](https://numaflow.numaproj.io/quick-start/#installation)

### Running the Simple Numalogic Pipeline

Once Numaflow is installed, create a simple Numalogic pipeline, which takes in time-series data, does the pre-processing, training, inference, and post-processing.

For building this pipeline, navigate to [numalogic-simple-pipeline](https://github.com/numaproj/numalogic/tree/main/examples/numalogic-simple-pipeline) under the examples folder and execute the following commands.

1. Build the docker image, import it to k3d, and apply the pipeline.
```shell
docker build -t numalogic-simple-pipeline:v1 . && k3d image import docker.io/library/numalogic-simple-pipeline:v1

kubectl apply -f numa-pl.yaml
```
2. To verify if the pipeline has been deployed successfully, check the status of each pod.
```shell
> kubectl get pods
NAME                                               READY   STATUS    RESTARTS   AGE
numaflow-server-d64bf6f7c-2czd7                    1/1     Running   0          72s
numaflow-controller-c84948cbb-994fn                1/1     Running   0          72s
isbsvc-default-js-0                                3/3     Running   0          68s
isbsvc-default-js-1                                3/3     Running   0          68s
isbsvc-default-js-2                                3/3     Running   0          68s
mlflow-sqlite-84cf5d6cd-pkmct                      1/1     Running   0          46s
numalogic-simple-pipeline-preprocess-0-mvuqb       2/2     Running   0          46s
numalogic-simple-pipeline-train-0-8xjg1            2/2     Running   0          46s
numalogic-simple-pipeline-daemon-66bbd94c4-hf4k2   1/1     Running   0          46s
numalogic-simple-pipeline-inference-0-n3asg        2/2     Running   0          46s
numalogic-simple-pipeline-postprocess-0-bw67q      2/2     Running   0          46s
numalogic-simple-pipeline-out-0-hjb7m              1/1     Running   0          46s
numalogic-simple-pipeline-in-0-tmd0v               1/1     Running   0          46s
```
### Sending data to the pipeline

Once the pipeline has been created, the data can be sent to the pipeline by port-forwarding the input vertex.

1. Port-forward to the http-source vertex
   ```shell
   kubectl port-forward simple-numalogic-pipeline-in-0-xxxxx 8443
   ```
   
2. Send the data to the pod via curl
   ```shell
   curl -kq -X POST https://localhost:8443/vertices/in -d '{"data":[0.9,0.1,0.2,0.9,0.9,0.9,0.9,0.8,1,0.9,0.9,0.7]}'
   ```
   Note: only send an array of length 12 in data, as the sequence length used for training is 12.   

   
### Training

Initially, there is no ML model present; to trigger training do a curl command and send any data to the pipeline. 

The training data is from [train_data.csv](https://github.com/numaproj/numalogic/blob/main/examples/numalogic-simple-pipeline/src/resources/train_data.csv), which follows a sinusoidal pattern where values fall in the range 200-350. 

The following logs will be seen in the training pod.

```shell
> curl -kq -X POST https://localhost:8443/vertices/in -d '{"data":[0.9,0.1,0.2,0.9,0.9,0.9,0.9,0.8,1,0.9,0.9,0.7]}'

> kubectl logs numalogic-simple-pipeline-train-0-xxxxx -c udf
2022-10-19 22:38:45,431 - INFO - Training autoencoder model..
2022-10-19 22:38:45,783 - INFO - epoch : 5, loss_mean : 0.0744678
2022-10-19 22:38:46,431 - INFO - epoch : 10, loss_mean : 0.0491540
...
2022-10-19 22:38:49,645 - INFO - epoch : 95, loss_mean : 0.0047888
2022-10-19 22:38:49,878 - INFO - epoch : 100, loss_mean : 0.0043651
2022-10-19 22:38:49,880 - INFO - 8b597791-b8a3-41b0-8375-47e168887c54 - Training complete
Successfully registered model 'ae::model'.
2022/10/19 22:38:52 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: ae::model, version 1
Created version '1' of model 'ae::model'.
2022-10-19 22:38:52,920 - INFO - 8b597791-b8a3-41b0-8375-47e168887c54 - Model Saving complete
```

### Inference

Now, the pipeline is ready for inference with the model trained above, data can be sent to the pipeline for ML inference. 

After sending the data, look for logs in the output pod, which shows the anomaly score.

Since we trained the model with data that follows a sinusoidal pattern where values range from 200-350, any value within this range is considered to be non-anomalous. And any value out of this range is considered to be anomalous.

Sending non-anomalous data: 
```
> curl -kq -X POST https://localhost:8443/vertices/in -d '{"data":[358.060687,326.253469,329.023996,346.168602,339.511273,359.080987,341.036110,333.584121,376.034150,351.065394,355.379422,333.347769]}'

> kubectl logs numalogic-simple-pipeline-out-0-xxxxx
2022/10/20 04:54:44 (out) {"ts_data": [[0.14472376660734326], [0.638373062689151], [0.8480656378656608], [0.4205087588581154], [1.285475729481929], [0.8136729095134241], [0.09972157219780131], [0.2856860200353754], [0.6005371351085002], [0.021966491476278518], [0.10405302543443251], [0.6428168263777302]], "anomaly_score": 0.49173648784304, "uuid": "0506b380-4565-405c-a3a3-ddc3a19e0bb4"}
```

Sending anomalous data:
```
> curl -kq -X POST https://localhost:8443/vertices/in -d '{"data":[358.060687,326.253469,329.023996,346.168602,339.511273,800.162220,614.091646,537.250124,776.034150,751.065394,700.379422,733.347769]}'

> kubectl logs numalogic-simple-pipeline-out-0-xxxxx
2022/10/20 04:56:40 (out) {"ts_data": [[1.173712319431301], [0.39061549013480673], [2.523849648503271], [2.0962927694957254], [13.032012667825805], [5.80166091013039], [3.6868855191928325], [4.814846700913904], [4.185973265627947], [3.9097889275446356], [4.505391607282856], [4.1170053183846305]], "anomaly_score": 3.9579276751803145, "uuid": "ed039779-f924-4801-9418-eeef30715ef1"}
```

In the output, `ts_data` is the final array that the input array has been transformed to, after all the steps in the pipeline. `anomaly_score` is the final anomaly score generated for the input data.


### MLflow UI

To see the model in MLflow UI, port forward mlflow-service using the below command and navigate to http://127.0.0.1:5000/
   ```shell
   kubectl port-forward svc/mlflow-service 5000
   ```


### Train on your own data
If you want to train an ML model on your own data, replace the `train_data.csv` file with your own file under [resources.](https://github.com/numaproj/numalogic/blob/main/examples/numalogic-simple-pipeline/src/resources) 

For more details, refer to [numalogic-simple-pipeline](https://github.com/numaproj/numalogic/tree/main/examples/numalogic-simple-pipeline) 





