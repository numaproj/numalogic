# Post Processing

Post-processing step is again an optional step, where we normalize the anomalies between 0-10. This is mostly to make the scores more understandable.

```python
import numpy as np
from numalogic.postprocess import tanh_norm

raw_anomaly_score = np.random.randn(10, 2)
test_anomaly_score_norm = tanh_norm(raw_anomaly_score)
```

A scikit-learn compatible API is also available.
```python
import numpy as np
from numalogic.postprocess import TanhNorm

raw_score = np.random.randn(10, 2)

norm = TanhNorm(scale_factor=10, smooth_factor=10)
norm_score = norm.fit_transform(raw_score)
```