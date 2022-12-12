# Threshold Estimators

Threshold Estimators are used for identifying the threshold limit above which we regard the datapoint as anomaly. 
It is a simple Estimator that extends BaseEstimator. 

Currently, the library supports `StdDevThreshold`. This takes in paramaters `min_thresh` and `std_factor`. This model 
defines threshold as `mean + 3 * std_factor`. 


Fitting the threshold model
```python
# preprocess step
clf = LogTransformer()
train_data = clf.fit_transform(X_train)
test_data = clf.transform(X_test)

# Fitting the Threshold model 
thresh_clf = StdDevThreshold(std_factor=1.2)
```

Train the model
```python
# Train the Autoencoder model and fit the model on train data
ae_pl = AutoencoderPipeline(
    model=Conv1dAE(in_channels=1, enc_channels=4), seq_len=8, num_epochs=30
)
ae_pl.fit(X_train)

# predict method returns the reconstruction error
anomaly_score = ae_pl.predict(X_test)
```
Predicting score using the threshold model
```python
# Predict final anomaly score using threshold estimator
anomaly_score = thresh_clf.predict(anomaly_score)
```