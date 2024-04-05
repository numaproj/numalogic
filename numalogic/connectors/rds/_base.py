from abc import ABCMeta, abstractmethod

import pandas as pd

from numalogic.connectors.rds._config import DatabaseServiceProvider, DatabaseTypes, RDSConfig
from numalogic.connectors.utils.aws.boto3_client_manager import Boto3ClientManager
from numalogic.connectors.utils.aws.exceptions import UnRecognizedDatabaseTypeException, \
    UnRecognizedDatabaseServiceProviderException



class RDSDataFetcher(object):

    def __init__(self, db_config: RDSConfig, **kwargs):
        self.kwargs = kwargs
        self.db_config = db_config
        self.connection = None
        self.database_type = db_config.database_type

    def get_password(self):
        db_password = None
        if self.db_config.aws_rds_use_iam or db_password == "":
            print("using aws_rds_use_iam ")
            boto3_client_manager = Boto3ClientManager(self.db_config)
            rds_client = boto3_client_manager.get_client(DatabaseServiceProvider.rds.value)
            db_password = boto3_client_manager.get_rds_token(rds_client)
        else:
            print("using password")
            db_password = self.db_config.database_password

        return db_password

    def get_connection(self):
        self.connection = ""
        pass

    def get_db_cursor(self):
        pass

    def execute_query(self, query) -> pd.DataFrame:
        pass
