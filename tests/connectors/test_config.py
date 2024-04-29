import pytest
from numalogic.connectors.exceptions import RDSFetcherConfValidationException
from numalogic.connectors._config import RDSFetcherConf


def test_RDSFetcherConf_post_init_exception():
    with pytest.raises(RDSFetcherConfValidationException):
        RDSFetcherConf(
            dimensions=[], metrics=[], datasource="test", hash_query_type=True, hash_column_name=""
        )
