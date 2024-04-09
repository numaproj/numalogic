from typing import Optional
import pandas as pd
from numalogic.connectors.utils.aws.config import DatabaseServiceProvider, RDSConfig
from numalogic.connectors.utils.aws.boto3_client_manager import Boto3ClientManager
import logging
from numalogic.connectors._config import Pivot
import time

_LOGGER = logging.getLogger(__name__)


class RDSDataFetcher:
    """
    Class: RDSDataFetcher.

    This class represents a data fetcher for RDS (Relational Database Service) connections. It
    provides methods for retrieving the RDS token, getting the password, establishing a
    connection, and executing queries.

    Attributes
    ----------
    - db_config (RDSConfig): The configuration object for the RDS connection.
    - kwargs (dict): Additional keyword arguments.

    Methods
    -------
    - get_rds_token(): Retrieves the RDS token using the Boto3ClientManager. - get_password() ->
    str: Retrieves the password for the RDS connection. If 'aws_rds_use_iam' is True, it calls
    the get_rds_token() method, otherwise it returns the database password from the
    configuration. - get_connection(): Placeholder method for establishing a connection to the
    RDS database. - get_db_cursor(): Placeholder method for getting a database cursor. -
    execute_query(query) -> pd.DataFrame: Placeholder method for executing a query and returning
    the result as a pandas DataFrame.

    """

    def __init__(self, db_config: RDSConfig, **kwargs):
        self.kwargs = kwargs
        self.db_config = db_config
        self.connection = None
        self.database_type = db_config.database_type
        self.boto3_client_manager = Boto3ClientManager(self.db_config)

    def get_rds_token(self) -> str:
        """
        Generates an RDS authentication token using the provided RDS boto3 client.

        Arguments
        ----------
        - rds_boto3_client (boto3.client): The RDS boto3 client used to generate the
        authentication token.

        Returns
        -------
            str: The generated RDS authentication token.

        """
        rds_client = self.boto3_client_manager.get_client(DatabaseServiceProvider.rds.value)
        return self.boto3_client_manager.get_rds_token(rds_client)

    def get_password(self) -> str:
        """
        Retrieves the password for the RDS connection.

        If 'aws_rds_use_iam' is True, it calls the get_rds_token() method to generate the RDS
        token. Otherwise, it returns the database password from the configuration.

        Returns
        -------
            str: The password for the RDS connection.

        """
        password = None
        if self.db_config.aws_rds_use_iam:
            _LOGGER.info("using aws_rds_use_iam to generate RDS Token")
            password = self.get_rds_token()
        else:
            _LOGGER.info("using password from config to connect RDS Database")
            password = self.db_config.database_password
        return password

    def get_connection(self):
        """
        Establishes a connection to the RDS database.

        This method is a placeholder and needs to be implemented in a subclass. It should handle
        the logic for establishing a connection to the RDS database based on the provided
        configuration.

        Returns
        -------
            None

        """
        raise NotImplementedError

    def get_db_cursor(self):
        """
        Retrieves a database cursor for executing queries.

        This method is a placeholder and needs to be implemented in a subclass. It should handle
        the logic for retrieving a database cursor based on the established connection.

        Returns
        -------
            None

        """
        pass

    def format_dataframe(
        self,
        df: pd.DataFrame,
        query: str,
        datetime_field_name: str,
        group_by: Optional[list[str]] = None,
        pivot: Optional[Pivot] = None,
    ):
        """
        Executes formatting operations on a pandas DataFrame.

        Arguments
        ----------
        df : pd.DataFrame
            The input DataFrame to be formatted.
        query : str
            The SQL query used to retrieve the data.
        datetime_field_name : str
            The name of the datetime field in the DataFrame.
        group_by : Optional[list[str]], optional
            A list of column names to group the DataFrame by, by default None.
        pivot : Optional[Pivot], optional
            An optional Pivot object specifying the index, columns,
            and values for pivoting the DataFrame, by default None.

        Returns
        -------
        pd.DataFrame : The formatted DataFrame.

        """
        _start_time = time.perf_counter()
        df["timestamp"] = pd.to_datetime(df[datetime_field_name]).astype("int64") // 10**6
        df.drop(columns=datetime_field_name, inplace=True)
        if group_by:
            df = df.groupby(by=group_by).sum().reset_index()

        if pivot and pivot.columns:
            df = df.pivot(
                index=pivot.index,
                columns=pivot.columns,
                values=pivot.value,
            )
            df.columns = df.columns.map("{0[1]}".format)
            df.reset_index(inplace=True)
        _end_time = time.perf_counter() - _start_time
        _LOGGER.info("RDS MYSQL Query: %s, Format time:  %.4fs", query, _end_time)
        return df

    def execute_query(self, query) -> pd.DataFrame:
        """
        Executes a query on the RDS database and returns the result as a pandas DataFrame.

        Parameters
        ----------
            query (str): The SQL query to be executed.

        Returns
        -------
            pd.DataFrame: The result of the query as a pandas DataFrame.

        """
        raise NotImplementedError
