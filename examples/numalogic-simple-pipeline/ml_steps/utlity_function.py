import json

import numpy as np
import pandas as pd
from numpy._typing import ArrayLike

from ml_steps.tools import MessagePacket


def _convert_df_to_json(df):
    json_df = df.to_json(orient="table")
    return json_df


def _convert_json_to_df(data):
    df = pd.read_json(data, orient="table")
    return df


def _convert_packet_payload(msg_packet: MessagePacket):
    json_df = _convert_df_to_json(msg_packet.df)
    data_packet = {
        "metric_name": msg_packet.metric_name,
        "data": json_df,
        "status": msg_packet.status,
        "anomaly_score": msg_packet.anomaly_score,
    }
    return data_packet


def _load_msg_packet(data: bytes):
    payload = json.loads(data.decode("utf-8"))
    msg_packet = MessagePacket(metric_name=payload["metric_name"])
    msg_packet.df = _convert_json_to_df(payload["data"])
    msg_packet.status = payload["status"]
    msg_packet.anomaly_score = payload["anomaly_score"]
    return msg_packet


def tanh_norm(scores: ArrayLike, scale_factor=10, smooth_factor=10) -> ArrayLike:
    return scale_factor * np.tanh(scores / smooth_factor)


def construct_key(skeys, dkeys) -> str:
    _static_key = ":".join(skeys)
    _dynamic_key = ":".join(dkeys)
    return "::".join([_static_key, _dynamic_key])
