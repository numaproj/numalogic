# Threshold Estimators

Threshold Estimators are used for identifying the threshold limit above which we regard the datapoint as anomaly.
It is a simple Estimator that extends BaseEstimator.

Currently, the library supports `StdDevThreshold`. This takes in paramaters `min_thresh` and `std_factor`. This model
defines threshold as `mean + 3 * std_factor`.

```python
import numpy as np
from numalogic.models.threshold import StdDevThreshold

# Generate positive random data
x_train = np.abs(np.random.randn(1000, 3))
x_test = np.abs(np.random.randn(30, 3))

# Here we want a threshold such that anything
# outside 5 deviations from the mean will be anomalous.
thresh_clf = StdDevThreshold(std_factor=5)
thresh_clf.fit(x_train)

# Let's get the predictions
y_pred = thresh_clf.predict(x_test)

# Anomaly scores can be given by, score_samples method
y_score = thresh_clf.score_samples(x_test)
```
