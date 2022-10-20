# Quick Start

## Installation

Install Numalogic and experiment with different tools available.

```shell
pip install numalogic
```

## Numalogic as a Library

Numalogic can be used as an independent library, it provides various ML models and tools. Here, we are using a `SparseAEPipeline`, you can refer to [training section](autoencoders.md) for other available options. 

In this example, we have numbers ranging from 1-10 in the train data set. In the test data set, we have data points that go out of this range, which the algorithm should be able to detect as anomalies.

```python
from numalogic.models.autoencoder import SparseAEPipeline
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.scores import tanh_norm

X_train = [[1], [3], [5], [2], [5], [1], [4], [5], [1], [4], [5], [8], [9], [1], [2], [4], [5], [1], [3]]
X_test = [[-20], [3], [5], [40], [5], [10], [4], [5], [100]]

model = SparseAEPipeline(
    model=Conv1dAE(in_channels=1, enc_channels=4), seq_len=8, num_epochs=30
)
# fit method trains the model on train data set
model.fit(X_train)

# predict method returns the reconstruction error
recon = model.predict(X_test)

# score method returns the anomaly score computed on test data set
anomaly_score = model.score(X_test)

# normalizing scores to range between 0-10
anomaly_score_norm = tanh_norm(anomaly_score)
print("Anomaly Scores:", anomaly_score_norm)
```

Below is the sample output, which has logs and anomaly scores printed. You can notice the anomaly score for points -20, 40 and 100 in `X_test` is high.
```shell
2022-10-19 17:37:08,241 - INFO - Current device: cpu
2022-10-19 17:37:08,241 - INFO - Current device: cpu
2022-10-19 17:37:08,245 - INFO - Training sparse autoencoder model with beta: 0.001, and rho: 0.05
2022-10-19 17:37:08,245 - INFO - Using kl_div regularized loss
2022-10-19 17:37:08,274 - INFO - epoch : 5, penalty: 0.00017940076941158623 loss_mean : 1.6290421
2022-10-19 17:37:08,282 - INFO - epoch : 10, penalty: 0.0001806353684514761 loss_mean : 1.5867264
2022-10-19 17:37:08,290 - INFO - epoch : 15, penalty: 0.00018205904052592814 loss_mean : 1.5468330
2022-10-19 17:37:08,297 - INFO - epoch : 20, penalty: 0.0001839457399910316 loss_mean : 1.5095336
2022-10-19 17:37:08,306 - INFO - epoch : 25, penalty: 0.00018673710292205215 loss_mean : 1.4750170
2022-10-19 17:37:08,314 - INFO - epoch : 30, penalty: 0.0001903115917230025 loss_mean : 1.4423900
Anomaly Scores: [[3.43350191]
 [1.0774759 ]
 [0.81873476]
 [5.34637364]
 [1.38783935]
 [0.3643268 ]
 [1.14278396]
 [2.78489574]
 [7.41677152]]
```

Replace `X_train` and `X_test` with your own data, and see the anomaly scores generated.

For more detailed experimentation, refer to [quick-start-example](https://github.com/numaproj/numalogic/blob/main/examples/quick-start.ipynb)

## Numalogic as streaming ML using Numaflow

Numalogic can also be paired with our streaming platform [Numaflow](https://numaflow.numaproj.io/), to build streaming ML pipelines where Numalogic can be used in UDF.

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



