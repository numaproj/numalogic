# Threshold Estimators

Threshold Estimators are used for identifying the threshold limit above which we regard the datapoint as anomaly. 
It is a simple Estimator that extends BaseEstimator. 

Currently, the library supports `StdDevThreshold`. This takes in paramaters `min_thresh` and `std_factor`. This model 
defines threshold as `mean + 3 * std_factor`. 


1. Fitting the threshold model
```python
thresh_clf = StdDevThreshold(std_factor=1.2)

thresh_clf.fit(train_data)
```


2. Predicting using the threshold model
```python
thresh_clf.predict(test_data)
```