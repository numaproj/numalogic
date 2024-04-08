import pandas as pd
from pydruid.utils import aggregators
from sklearn.pipeline import make_pipeline
from numalogic.config import PreprocessFactory
from numalogic.connectors import DruidFetcher
from numalogic.connectors._config import Pivot
from numalogic.udfs import TrainerUDF
import numpy as np


class ModifiedDruidFetcher(DruidFetcher):
    def __init__(self, url: str, endpoint: str, file: str):
        super().__init__(url, endpoint)
        self.file =file
    def _fetch(self,**query_params) -> pd.DataFrame:
        return pd.read_csv(self.file, index_col=None)
class ModifiedDruidTrainerUDF(TrainerUDF):
    def __init__(self,r_client, pl_conf, file):
        super().__init__(r_client=r_client, pl_conf=pl_conf)
        self.dataconn_conf = self.pl_conf.druid_conf
        self.data_fetcher = ModifiedDruidFetcher(
            url=self.dataconn_conf.url, endpoint=self.dataconn_conf.endpoint, file=file)


def get_feature_arr(
        raw_df,
        metrics: list[str],
        fill_value: float = 0.0,
):
    """
    Get feature array from the raw dataframe.

    Args:
        raw_df: Raw dataframe
        metrics: List of metrics
        fill_value: Value to fill missing values with

    Returns
    -------
        Numpy array
        nan_counter: Number of nan values
        inf_counter: Number of inf values
    """
    nan_counter = 0
    for col in metrics:
        if col not in raw_df.columns:
            raw_df[col] = fill_value
            nan_counter += len(raw_df)
    feat_df = raw_df[metrics]
    nan_counter += raw_df.isna().sum().all()
    inf_counter = np.isinf(feat_df).sum().all()
    feat_df = feat_df.fillna(fill_value).replace([np.inf, -np.inf], fill_value)
    return feat_df.to_numpy(dtype=np.float32), nan_counter, inf_counter

def _construct_clf(_conf):
    preproc_clfs = []
    if not _conf:
        return None
    for _cfg in _conf:
        _clf = PreprocessFactory().get_instance(_cfg)
        preproc_clfs.append(_clf)
    if not preproc_clfs:
        return None
    if len(preproc_clfs) == 1:
        return preproc_clfs[0]
    return make_pipeline(*preproc_clfs)


def data_fetch(filter,keys,file_name):
    fetcher = ModifiedDruidFetcher("lol", "druid/v2",file_name)
    return fetcher.fetch(
        filter_keys=filter,
        filter_values=keys,
        dimensions=["ciStatus"],
        datasource="tech-ip-customer-interaction-metrics",
        aggregations={"count": aggregators.doublesum("count")},
        group_by=["timestamp", "ciStatus"],
        hours=1,
        pivot=Pivot(
            index="timestamp",
            columns=["ciStatus"],
            value=["count"],
        ),
    )