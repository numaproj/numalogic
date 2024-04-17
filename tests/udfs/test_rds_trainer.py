import pytest
from numalogic.tools.exceptions import ConfigNotFoundError, RDSFetcherError
from numalogic.udfs._config import PipelineConf
from numalogic.connectors import RDSFetcherConf, RDSFetcher
from numalogic.udfs.trainer._rds import (RDSTrainerUDF, build_query, get_hash_based_query)
import pytz
import re
from datetime import datetime
from numalogic._constants import TESTS_DIR
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs._config import load_pipeline_conf
from unittest.mock import patch, Mock
from numalogic.connectors._config import Pivot
from typing import Optional
import pandas as pd
import os

# @pytest.fixture
def mock_rds_fetch_data(
        query,
        datetime_column_name: str,
        pivot: Optional[Pivot] = None,
        group_by: Optional[list[str]] = None,
        ):
    nrows=5000
    """Mock druid fetch data."""
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "rds.csv"),
        index_col="timestamp",
        nrows=nrows,
    )


@pytest.fixture()
def mock_trainer_payload():
    trainer_payload = TrainerPayload(
        uuid="979-98789798-98787",
        config_id="fciPluginAppInteractions",
        pipeline_id="metrics",
        composite_keys=["pluginAssetId", "assetId", "interactionName"],
        metrics=["failed", "degraded"]
    )
    return trainer_payload


@pytest.fixture()
def mock_redis_client():
    return FakeStrictRedis(server=FakeServer())


@pytest.fixture()
def mock_pipeline_conf():
    pipeline_conf = f"{TESTS_DIR}/resources/rds_trainer_config_fetcher_conf.yaml"
    return load_pipeline_conf(pipeline_conf)

@pytest.fixture()
def mock_RDS_trainer_UDF(mock_pipeline_conf):
    return RDSTrainerUDF(mock_redis_client, mock_pipeline_conf)


def test_get_hash_based_query():
    # assumption of correct functionality
    assert get_hash_based_query('123', ['foo'], ['bar']) == '88b8e77c0dc4fe0bbdea9250b6aa4705'

    # assumption of behavior when lists are not of equal length
    with pytest.raises(RDSFetcherError):
        get_hash_based_query('123', ['foo'], ['bar', 'baz'])


def test_build_query(mock_trainer_payload, mock_pipeline_conf):
    # assumption of correct functionality
    datetime_str = '04/23/24 13:55:26'
    test_time = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
    query = build_query('foo', True, '123', ['a', 'b'], ['c', 'd'], ['a'], ['b'], 'time', 'hash',
                        2.0, 1.0, reference_dt=test_time).replace("\n", " ")
    actual_query = re.sub(r'\s+', ' ', query)

    expected_query = "select time, a, b, c, d from foo where time >= '2024-04-23T10:55:26' and time <= '2024-04-23T12:55:26' and hash = '1692fcfff3e01e7ba8cffc2baadef5f5'"

    print(actual_query)
    assert actual_query.strip() == expected_query

    # assumption of behavior when hash_query_type is False
    with pytest.raises(RDSFetcherError):
        build_query('foo', False, '123', ['a', 'b'], ['c', 'd'], ['a'], ['b'], 'time', 'hash', 2.0,
                    1.0)


# @patch.object(RDSFetcher, "fetch", Mock(return_value=mock_rds_fetch_data))
def test_rds_trainer(mock_trainer_payload, mock_pipeline_conf, mock_RDS_trainer_UDF):
    with patch.object(mock_RDS_trainer_UDF.data_fetcher, 'fetch', new=mock_rds_fetch_data):
        actual_df = mock_RDS_trainer_UDF.fetch_data(mock_trainer_payload)
        actual_df_count_dict = actual_df.count().to_dict()
        expected_df_count_dict = {'degraded': 4986, 'failed': 5000, 'success': 5000}
        assert actual_df_count_dict == expected_df_count_dict

