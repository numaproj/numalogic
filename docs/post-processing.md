# Post Processing

Post-processing step is again an optional step, where we normalize the anomalies between 0-10. This is mostly to make the scores more understandable.

```python
from numalogic.scores import tanh_norm
test_anomaly_score_norm = tanh_norm(test_anomaly_score)
```