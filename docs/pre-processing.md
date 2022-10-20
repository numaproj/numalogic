# Pre Processing

When creating a Machine Learning pipeline, data pre-processing plays a crucial role that takes in raw data and transforms it into a format that can be understood and analyzed by the ML Models.

Generally, the majority of real-word datasets are incomplete, inconsistent or inaccurate (contains errors or outliers). Applying ML algorithms on this raw data would give inaccurate results, as they would fail to identify the underlying patterns effectively.

Quality decisions must be based on quality data. Data Preprocessing is important to get this quality data, without which it would just be a Garbage In, Garbage Out scenario.

Numalogic provides the following tranformers for pre-processing the training or testing data sets. You can also pair it with scalers like `MinMaxScaler` from [scikit-learn pre-processing](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing) tools.

### Log Transformer

Log transformation is a data transformation method in which it replaces each data point x with a log(x). `LogTransformer` reduces the variance in some distributions, especially with large outliers.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from numalogic.preprocess.transformer import LogTransformer

transformer = LogTransformer(add_factor=1)
scaler = MinMaxScaler()

pipeline = make_pipeline(transformer, scaler)

X_train = transformer.transform(train_df.to_numpy())
X_test = scaler.transform(test_df.to_numpy())
```

### Static Power Transformer

Static Power Transformer converts each data point x to x<sup>n</sup>. 

```python
from numalogic.preprocess.transformer import StaticPowerTransformer

transformer = StaticPowerTransformer(n=3, add_factor=2)
scaler = MinMaxScaler()

pipeline = make_pipeline(transformer, scaler)

X_train = transformer.transform(train_df.to_numpy())
X_test = scaler.transform(test_df.to_numpy())
```
