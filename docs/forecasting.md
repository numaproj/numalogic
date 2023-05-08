# Forcasting

Numalogic supports the following variants of forecasting based anomaly detection models.

## Naive Forecasters

### Baseline Forecaster

This is a naive forecaster, that uses a combination of:

1. Log transformation
2. Z-Score normalization

```python
from numalogic.models.forecast.variants import BaselineForecaster

model = BaselineForecaster()
model.fit(train_df)
pred_df = model.predict(test_df)
r2_score = model.r2_score(test_df)
anomaly_score = model.score(test_df)
```
### Seasonal Naive Forecaster

A naive forecaster that takes seasonality into consideration and predicts the previous day/week values.

```python
from numalogic.models.forecast.variants import SeasonalNaiveForecaster

model = SeasonalNaiveForecaster()
model.fit(train_df)
pred_df = model.predict(test_df)
r2_score = model.r2_score(test_df)
anomaly_score = model.score(test_df)
```
