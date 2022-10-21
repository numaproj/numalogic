# Numalogic Pipeline

## Running Example: Quickstart Numalogic Pipeline

### Pre-requisite setup
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
3. Wait for the pipeline to be up
```
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

Initially there is no ML model present. So send curl command with any data to train the model. the following logs will come to the training pod
```
2022-10-19 22:38:45,431 - INFO - Training autoencoder model..
2022-10-19 22:38:45,783 - INFO - epoch : 5, loss_mean : 0.0744678
2022-10-19 22:38:46,431 - INFO - epoch : 10, loss_mean : 0.0491540
2022-10-19 22:38:46,588 - INFO - epoch : 15, loss_mean : 0.0441010
2022-10-19 22:38:46,926 - INFO - epoch : 20, loss_mean : 0.0331293
2022-10-19 22:38:47,139 - INFO - epoch : 25, loss_mean : 0.0314313
2022-10-19 22:38:47,300 - INFO - epoch : 30, loss_mean : 0.0244382
2022-10-19 22:38:47,475 - INFO - epoch : 35, loss_mean : 0.0238161
2022-10-19 22:38:47,593 - INFO - epoch : 40, loss_mean : 0.0183135
2022-10-19 22:38:47,730 - INFO - epoch : 45, loss_mean : 0.0150381
2022-10-19 22:38:47,847 - INFO - epoch : 50, loss_mean : 0.0138239
2022-10-19 22:38:48,078 - INFO - epoch : 55, loss_mean : 0.0111972
2022-10-19 22:38:48,321 - INFO - epoch : 60, loss_mean : 0.0096366
2022-10-19 22:38:48,535 - INFO - epoch : 65, loss_mean : 0.0087036
2022-10-19 22:38:48,716 - INFO - epoch : 70, loss_mean : 0.0078248
2022-10-19 22:38:48,908 - INFO - epoch : 75, loss_mean : 0.0070344
2022-10-19 22:38:49,098 - INFO - epoch : 80, loss_mean : 0.0060536
2022-10-19 22:38:49,306 - INFO - epoch : 85, loss_mean : 0.0052490
2022-10-19 22:38:49,481 - INFO - epoch : 90, loss_mean : 0.0053800
2022-10-19 22:38:49,645 - INFO - epoch : 95, loss_mean : 0.0047888
2022-10-19 22:38:49,878 - INFO - epoch : 100, loss_mean : 0.0043651
2022-10-19 22:38:49,880 - INFO - 8b597791-b8a3-41b0-8375-47e168887c54 - Training complete
Successfully registered model 'ae::model'.
2022/10/19 22:38:52 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: ae::model, version 1
Created version '1' of model 'ae::model'.
2022-10-19 22:38:52,920 - INFO - 8b597791-b8a3-41b0-8375-47e168887c54 - Model Saving complete
```



Now if we send non-anomalous data: 
```
curl -kq -X POST https://localhost:8443/vertices/in -d '{"data":[358.060687,326.253469,329.023996,346.168602,339.511273,359.080987,341.036110,333.584121,376.034150,351.065394,355.379422,333.347769]}'

2022/10/20 04:54:44 (out) {"ts_data": [[0.14472376660734326], [0.638373062689151], [0.8480656378656608], [0.4205087588581154], [1.285475729481929], [0.8136729095134241], [0.09972157219780131], [0.2856860200353754], [0.6005371351085002], [0.021966491476278518], [0.10405302543443251], [0.6428168263777302]], "anomaly_score": 0.49173648784304, "uuid": "0506b380-4565-405c-a3a3-ddc3a19e0bb4"}
```

Now if we send anomalous data:
```
curl -kq -X POST https://localhost:8443/vertices/in -d '{"data":[358.060687,326.253469,329.023996,346.168602,339.511273,800.162220,614.091646,537.250124,776.034150,751.065394,700.379422,733.347769]}'


2022/10/20 04:56:40 (out) {"ts_data": [[1.173712319431301], [0.39061549013480673], [2.523849648503271], [2.0962927694957254], [13.032012667825805], [5.80166091013039], [3.6868855191928325], [4.814846700913904], [4.185973265627947], [3.9097889275446356], [4.505391607282856], [4.1170053183846305]], "anomaly_score": 3.9579276751803145, "uuid": "ed039779-f924-4801-9418-eeef30715ef1"}
```