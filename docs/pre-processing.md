# Pre Processing

When creating a Machine Learning pipeline, data pre-processing plays a crucial role that takes in raw data and transforms it into a format that can be understood and analyzed by the ML Models.

Generally, the majority of real-word datasets are incomplete, inconsistent or inaccurate (contains errors or outliers). Applying ML algorithms on this raw data would give inaccurate results, as they would fail to identify the underlying patterns effectively.

Quality decisions must be based on quality data. Data Preprocessing is important to get this quality data, without which it would just be a Garbage In, Garbage Out scenario.

Numalogic provides the following tranformers for pre-processing the training or testing data sets. You can also pair it with scalers like `MinMaxScaler` from [scikit-learn pre-processing](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing) tools.

### Log Transformer

Log transformation is a data transformation method in which it replaces each data point x with a log(x).

Now, with `add_factor`, each data point x is converted to log(x + add_factor)

Log transformation reduces the variance in some distributions, especially with large outliers.

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from numalogic.transforms import LogTransformer

# Generate some random train and test data
x_train = np.random.randn(100, 3)
x_test = np.random.randn(20, 3)

transformer = LogTransformer(add_factor=4)
scaler = MinMaxScaler()

pipeline = make_pipeline(transformer, scaler)

x_train_scaled = pipeline.fit_transform(x_train)
X_test_scaled = pipeline.transform(x_test)
```

### Static Power Transformer

Static Power Transformer converts each data point x to x<sup>n</sup>.

When `add_factor` is provided, each data point x is converted to (x + add_factor)<sup>n</sup>

```python
import numpy as np
from numalogic.transforms import StaticPowerTransformer

# Generate some random train and test data
x_train = np.random.randn(100, 3)
x_test = np.random.randn(20, 3)

transformer = StaticPowerTransformer(n=3, add_factor=2)

# Since this transformer is stateless, we can just call transform()
x_train_scaled = transformer.transform(x_train)
X_test_scaled = transformer.transform(x_test)
```

### Tanh Scaler

Tanh Scaler is a stateful estimator that applies tanh normalization to the Z-score,
and scales the values between 0 and 1.
This scaler is seen to be more efficient as well as robust to the effect of outliers
in the data.

```python
import numpy as np
from numalogic.transforms import TanhScaler

# Generate some random train and test data
x_train = np.random.randn(100, 3)
x_test = np.random.randn(20, 3)

scaler = TanhScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```
