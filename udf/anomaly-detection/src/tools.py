import time

import numpy as np
import pytz
import socket

import pandas as pd
from typing import List
from functools import wraps
from json import JSONDecodeError
from collections import OrderedDict
from datetime import timedelta, datetime

from numalogic.config import PostprocessFactory, ModelInfo
from numalogic.models.threshold import SigmoidThreshold

from src import get_logger, MetricConf
from src._config import StaticThreshold
from src.connectors.prometheus import Prometheus
from src.entities import TrainerPayload, Matrix
from src.watcher import ConfigManager

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


def calculate_static_thresh(x_arr: Matrix, static_threshold: StaticThreshold) -> np.ndarray:
    """
    Calculates anomaly scores using static thresholding.
    """
    static_clf = SigmoidThreshold(upper_limit=static_threshold.upper_limit)
    static_scores = static_clf.score_samples(x_arr)
    return static_scores


class WindowScorer:
    """
    Class to calculate the final anomaly scores for the window.

    Args:
        static_thresh: StaticThreshold instance
        postprocess_conf: ModelInfo instance
    """

    __slots__ = ("static_thresh", "model_wt", "postproc_clf")

    def __init__(self, static_thresh: StaticThreshold, postprocess_conf: ModelInfo):
        self.static_thresh = static_thresh
        self.model_wt = 1.0 - self.static_thresh.weight
        postproc_factory = PostprocessFactory()
        self.postproc_clf = postproc_factory.get_instance(postprocess_conf)

    def get_ensemble_score(self, x_arr: Matrix) -> float:
        """
        Returns the final normalized window score.

        Performs soft voting ensembling if valid static threshold
        weight found in config.

        Args:
            x_arr: Metric scores array

        Returns:
            Final score for the window
        """
        norm_score = self.get_norm_score(x_arr)

        if not self.static_thresh.weight:
            return norm_score

        norm_static_score = self.get_static_score(x_arr)
        ensemble_score = (self.static_thresh.weight * norm_static_score) + (
            self.model_wt * norm_score
        )
        return ensemble_score

    def get_static_score(self, x_arr) -> float:
        """
        Returns the normalized window score
        calculated using the static threshold estimator.

        Args:
            x_arr: Metric scores array

        Returns:
            Score for the window
        """
        static_scores = calculate_static_thresh(x_arr, self.static_thresh)
        static_score = np.mean(static_scores)
        return self.postproc_clf.transform(static_score)

    def get_norm_score(self, x_arr: Matrix):
        """
        Returns the normalized window score

        Args:
            x_arr: Metric scores array

        Returns:
            Score for the window
        """
        win_score = np.mean(x_arr)
        return self.postproc_clf.transform(win_score)

    def adjust_weights(self):
        """
        Adjust the soft voting weights depending on the streaming input.
        """
        raise NotImplementedError
