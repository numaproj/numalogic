# Post Processing
After the raw scores have been generated, we might need to do some additional postprocessing,
for various reasons.

### Tanh Score Normalization
Tanh normalization step is an optional step, where we normalize the anomalies between 0-10. This is mostly to make the scores more understandable.

```python
import numpy as np
from numalogic.transforms import tanh_norm

raw_anomaly_score = np.random.randn(10, 2)
test_anomaly_score_norm = tanh_norm(raw_anomaly_score)
```

A scikit-learn compatible API is also available.

```python
import numpy as np
from numalogic.transforms import TanhNorm

raw_score = np.random.randn(10, 2)

norm = TanhNorm(scale_factor=10, smooth_factor=10)
norm_score = norm.fit_transform(raw_score)
```

### Exponentially Weighted Moving Average
The Exponentially Weighted Moving Average (EWMA) serves as an effective smoothing function,
emphasizing the importance of more recent anomaly scores over those of previous elements within a sliding window.

This approach proves particularly beneficial in streaming inference scenarios, as it allows for
earlier increases in anomaly scores when a new outlier data point is encountered.
Consequently, the EMA enables a more responsive and dynamic assessment of streaming data,
facilitating timely detection and response to potential anomalies.

```python
import numpy as np
from numalogic.transforms import ExpMovingAverage

raw_score = np.array([1.0, 1.5, 1.2, 3.5, 2.7, 5.6, 7.1, 6.9, 4.2, 1.1]).reshape(-1, 1)

postproc_clf = ExpMovingAverage(beta=0.5)
out = postproc_clf.transform(raw_score)

# out: [[1.3], [1.433], [1.333], [2.473], [2.591], [4.119], [5.621], [6.263], [5.229], [3.163]]
```
