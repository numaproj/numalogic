import socket
import time
from collections import OrderedDict
from datetime import timedelta, datetime
from functools import wraps
from json import JSONDecodeError
from typing import Optional, Sequence, List

import boto3
import numpy as np
import pandas as pd
import pytz
from botocore.session import get_session
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from numalogic.config import PostprocessFactory
from numalogic.models.threshold import SigmoidThreshold
from numalogic.registry import MLflowRegistry, ArtifactData
from pynumaflow.function import Messages, Message

from anomalydetection import get_logger, MetricConf
from anomalydetection.entities import TrainerPayload, StreamPayload
from anomalydetection.clients.prometheus import Prometheus
from anomalydetection.watcher import ConfigManager

_LOGGER = get_logger(__name__)


def catch_exception(func):
    @wraps(func)
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except JSONDecodeError as err:
            _LOGGER.exception("Error in json decode for %s: %r", func.__name__, err)
        except Exception as ex:
            _LOGGER.exception("Error in %s: %r", func.__name__, ex)

    return inner_function


def msgs_forward(handler_func):
    @wraps(handler_func)
    def inner_function(*args, **kwargs):
        json_list = handler_func(*args, **kwargs)
        msgs = Messages()
        for json_data in json_list:
            if json_data:
                msgs.append(Message.to_all(json_data))
            else:
                msgs.append(Message.to_drop())
        return msgs

    return inner_function


def msg_forward(handler_func):
    @wraps(handler_func)
    def inner_function(*args, **kwargs):
        json_data = handler_func(*args, **kwargs)
        msgs = Messages()
        if json_data:
            msgs.append(Message.to_all(value=json_data))
        else:
            msgs.append(Message.to_drop())
        return msgs

    return inner_function


def conditional_forward(hand_func):
    @wraps(hand_func)
    def inner_function(*args, **kwargs) -> Messages:
        data = hand_func(*args, **kwargs)
        msgs = Messages()
        for vertex, json_data in data:
            if json_data and vertex:
                msgs.append(Message.to_vtx(key=vertex.encode(), value=json_data))
            else:
                msgs.append(Message.to_drop())
        return msgs

    return inner_function


def create_composite_keys(msg: dict, keys: List[str]) -> OrderedDict:
    labels = msg.get("labels")
    result = OrderedDict()
    for k in keys:
        if k in msg:
            result[k] = msg[k]
        if k in labels:
            result[k] = labels[k]
    return result


def get_ipv4_by_hostname(hostname: str, port=0) -> list:
    return list(
        idx[4][0]
        for idx in socket.getaddrinfo(hostname, port)
        if idx[0] is socket.AddressFamily.AF_INET and idx[1] is socket.SocketKind.SOCK_RAW
    )


def is_host_reachable(hostname: str, port=None, max_retries=5, sleep_sec=5) -> bool:
    retries = 0
    assert max_retries >= 1, "Max retries has to be at least 1"

    while retries < max_retries:
        try:
            get_ipv4_by_hostname(hostname, port)
        except socket.gaierror as ex:
            retries += 1
            _LOGGER.warning(
                "Failed to resolve hostname: %s: error: %r", hostname, ex, exc_info=True
            )
            time.sleep(sleep_sec)
        else:
            return True
    _LOGGER.error("Failed to resolve hostname: %s even after retries!")
    return False


def load_model(
    skeys: Sequence[str], dkeys: Sequence[str], artifact_type: str = "pytorch"
) -> Optional[ArtifactData]:
    set_aws_session()
    try:
        registry_conf = ConfigManager.get_registry_config()
        ml_registry = MLflowRegistry(
            tracking_uri=registry_conf.tracking_uri, artifact_type=artifact_type
        )
        return ml_registry.load(skeys=skeys, dkeys=dkeys)
    except RestException as warn:
        if warn.error_code == 404:
            return None
        _LOGGER.warning("Non 404 error from mlflow: %r", warn)
    except Exception as ex:
        _LOGGER.error("Unexpected error while loading model from MLflow database: %r", ex)
        return None


def save_model(
    skeys: Sequence[str], dkeys: Sequence[str], model, artifact_type="pytorch", **metadata
) -> Optional[ModelVersion]:
    set_aws_session()
    registry_conf = ConfigManager.get_registry_config()
    ml_registry = MLflowRegistry(
        tracking_uri=registry_conf.tracking_uri, artifact_type=artifact_type
    )
    version = ml_registry.save(skeys=skeys, dkeys=dkeys, artifact=model, **metadata)
    return version


def fetch_data(
    payload: TrainerPayload,
    metric_config: MetricConf,
    labels: dict,
    return_labels=None,
    hours: int = 36,
) -> pd.DataFrame:
    _start_time = time.time()
    prometheus_conf = ConfigManager.get_prometheus_config()
    datafetcher = Prometheus(prometheus_conf.server)

    end_dt = datetime.now(pytz.utc)
    start_dt = end_dt - timedelta(hours=hours)

    df = datafetcher.query_metric(
        metric_name=payload.composite_keys["name"],
        labels_map=labels,
        return_labels=return_labels,
        start=start_dt.timestamp(),
        end=end_dt.timestamp(),
        step=metric_config.scrape_interval,
    )
    _LOGGER.info(
        "%s - Time taken to fetch data: %s, for df shape: %s",
        payload.uuid,
        time.time() - _start_time,
        df.shape,
    )
    return df


def set_aws_session() -> None:
    """
    Setup default aws session by refreshing credentials.
    """
    session = get_session()
    credentials = session.get_credentials()
    if not credentials:
        _LOGGER.debug("No AWS credentials object returned")
        return
    boto3.setup_default_session(
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        aws_session_token=credentials.token,
    )


def calculate_static_thresh(payload: StreamPayload, upper_limit: float):
    """
    Calculates anomaly scores using static thresholding.
    """
    x = payload.get_stream_array(original=True)
    static_clf = SigmoidThreshold(upper_limit=upper_limit)
    static_scores = static_clf.score_samples(x)
    return static_scores


class WindowScorer:
    """
    Class to calculate the final anomaly scores for the window.

    Args:
        metric_conf: MetricConf instance
    """

    __slots__ = ("static_wt", "static_limit", "model_wt", "postproc_clf")

    def __init__(self, metric_conf: MetricConf):
        self.static_wt = metric_conf.static_threshold_wt
        self.static_limit = metric_conf.static_threshold
        self.model_wt = 1.0 - self.static_wt

        postproc_factory = PostprocessFactory()
        self.postproc_clf = postproc_factory.get_instance(metric_conf.numalogic_conf.postprocess)

    def get_final_winscore(self, payload: StreamPayload) -> float:
        """
        Returns the final normalized window score.

        Performs soft voting ensembling if valid static threshold
        weight found in config.

        Args:
            payload: StreamPayload instance

        Returns:
            Final score for the window
        """
        norm_winscore = self.get_winscore(payload)

        if not self.static_wt:
            return norm_winscore

        norm_static_winscore = self.get_static_winscore(payload)
        ensemble_score = (self.static_wt * norm_static_winscore) + (self.model_wt * norm_winscore)

        _LOGGER.debug(
            "%s - Model score: %s, Static score: %s, Static wt: %s",
            payload.uuid,
            norm_winscore,
            norm_static_winscore,
            self.static_wt,
        )

        return ensemble_score

    def get_static_winscore(self, payload: StreamPayload) -> float:
        """
        Returns the normalized window score
        calculated using the static threshold estimator.

        Args:
            payload: StreamPayload instance

        Returns:
            Score for the window
        """
        static_scores = calculate_static_thresh(payload, self.static_limit)
        static_winscore = np.mean(static_scores)
        return self.postproc_clf.transform(static_winscore)

    def get_winscore(self, payload: StreamPayload):
        """
        Returns the normalized window score

        Args:
            payload: StreamPayload instance

        Returns:
            Score for the window
        """
        scores = payload.get_stream_array()
        winscore = np.mean(scores)
        return self.postproc_clf.transform(winscore)

    def adjust_weights(self):
        """
        Adjust the soft voting weights depending on the streaming input.
        """
        raise NotImplementedError
