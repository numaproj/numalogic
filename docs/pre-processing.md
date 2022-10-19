# Pre Processing

When creating a Machine Learning pipeline, data pre-processing plays a crucial role that takes in raw data and transforms it into a format that can be understood and analyzed by the ML Models.

Generally, the majority of real-word datasets are incomplete, inconsistent or inaccurate (contains errors or outliers). Applying ML algorithms on this raw data would give inaccurate results, as they would fail to identify the underlying patterns effectively.

Quality decisions must be based on quality data. Data Preprocessing is important to get this quality data, without which it would just be a Garbage In, Garbage Out scenario.

Let us see how we can clean the data before we do the training/inference.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df.to_numpy())
X_test = scaler.transform(test_df.to_numpy())
```