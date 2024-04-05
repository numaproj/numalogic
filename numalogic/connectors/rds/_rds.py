from numalogic.connectors.rds import db
from numalogic.connectors.rds._config import DatabaseTypes, RDSConfig
from numalogic.connectors.utils.aws.db_configurations import load_db_conf
import logging
import pandas as pd

_LOGGER = logging.getLogger(__name__)



class RDSFetcher(object):

    def __init__(self, db_config: RDSConfig):
        self.db_config = db_config
        if db.CLASS_TYPE:
            self.fetcher = db.CLASS_TYPE(db_config)
            print(self.fetcher.database_type)

    def fetch(self, query) -> pd.DataFrame:
        return self.fetcher.execute_query(query)

#
# if __name__ == "__main__":
#     pd.options.display.max_columns = 1000
#     pd.options.display.max_rows = 1000
#     config = load_db_conf(
#         "/Users/skondakindi/Desktop/codebase/odl/odl-ml-python-sdk/tests/resources/db_config_no_ssl.yaml"
#     )
#     rds = RDSFetcher(config)
#     result = rds.fetch("select 1")
#     _LOGGER.info(result.to_json(orient="records"))
