import json
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pynumaflow.function import Messages, Message, Datum

from ml_steps.tools import Status
from ml_steps.utlity_function import _convert_df_to_json

DEFAULT_WIN_SIZE = 12


def input(key: str, datum: Datum) -> Messages:
    val = datum.value
    _ = datum.event_time
    _ = datum.watermark
    times = pd.date_range(datetime.now(), periods=DEFAULT_WIN_SIZE, freq="5min")
    data = []
    for i in range(DEFAULT_WIN_SIZE):
        data.append([times[i].timestamp(), float(np.random.rand())])
    metric_list = [
        "error_rate",
        "error_count",
    ]
    metric_name = metric_list[random.randint(0, 1)]
    messages = Messages()

    # Have data in Numpy array -> DataFrame -> Json
    df = pd.DataFrame(data=data, columns=["Timestamp", "Value"])
    df.set_index("Timestamp", inplace=True)
    json_df = _convert_df_to_json(df)

    msg_packet = {
        "metric_name": metric_name,
        "data": json_df,
        "status": Status.RAW.value,
        "anomaly_score": 0.0,
    }
    time.sleep(1)
    print(json.dumps(msg_packet).encode("utf-8"))
    messages.append(Message.to_all(json.dumps(msg_packet).encode("utf-8")))
    return messages
