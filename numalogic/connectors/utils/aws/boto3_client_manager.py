import boto3
from boto3 import Session
import logging

from numalogic.connectors.utils.aws.config import DatabaseServiceProvider, RDSConnectionConfig
from numalogic.connectors.utils.aws.exceptions import UnRecognizedAWSClientException
from numalogic.connectors.utils.aws.sts_client_manager import STSClientManager

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


class Boto3ClientManager:
    """
    Class: Boto3ClientManager.

    The 'Boto3ClientManager' class is responsible for managing AWS clients for different
    services like RDS and Athena. It uses the provided configurations to create the clients and
    manage their sessions.

    Methods
    -------
    __init__(self, configurations): Initializes the Boto3ClientManager with the given
    configurations. - get_boto3_session(self) -> Session: Returns a Boto3 session object with
    the necessary credentials. - get_rds_token(self, rds_boto3_client) -> str: Generates an RDS
    authentication token using the provided RDS boto3 client. - get_client(self, client_type:
    str): Generates an AWS client based on the provided client type.

    Note: This class should be instantiated with the necessary configurations before using its
    methods.
    """

    def __init__(self, configurations: RDSConnectionConfig):
        self.rds_client = None
        self.athena_client = None
        self.configurations = configurations
        self.sts_client_manager = STSClientManager()

    def get_boto3_session(self) -> Session:
        """
        Returns a Boto3 session object with the necessary credentials.

        This method retrieves the credentials from the STSClientManager using the given AWS
        assume role ARN and session name. It then creates a Boto3 session object with the
        retrieved credentials and returns it.

        Returns
        -------
            Session: A Boto3 session object with the necessary credentials.

        """
        credentials = self.sts_client_manager.get_credentials(
            self.configurations.aws_assume_role_arn,
            self.configurations.aws_assume_role_session_name,
        )
        tmp_access_key = credentials["AccessKeyId"]
        tmp_secret_key = credentials["SecretAccessKey"]
        security_token = credentials["SessionToken"]
        return Session(
            aws_access_key_id=tmp_access_key,
            aws_secret_access_key=tmp_secret_key,
            aws_session_token=security_token,
        )

    def get_rds_token(self, rds_boto3_client: boto3.session.Session.client) -> str:
        """
        Generates an RDS authentication token using the provided RDS boto3 client.

        This method generates an RDS authentication token by calling the
        'generate_db_auth_token' method of the provided RDS boto3 client. The authentication
        token is generated using the following parameters: - DBHostname: The endpoint of the RDS
        database. - Port: The port number of the RDS database. - DBUsername: The username for
        the RDS database. - Region: The AWS region where the RDS database is located.

        Args:
             - rds_boto3_client (boto3.client): The RDS boto3 client used to generate the
             authentication token.

        Returns
        -------
            str: The generated RDS authentication token.

        """
        return rds_boto3_client.generate_db_auth_token(
            DBHostname=self.configurations.endpoint,
            Port=self.configurations.port,
            DBUsername=self.configurations.database_username,
            Region=self.configurations.aws_region,
        )

    def get_client(self, client_type: str) -> boto3.session.Session.client:
        """
        Generates an AWS client based on the provided client type.

        This method generates an AWS client based on the provided client type. It first checks
        if the client type is recognized by checking if it exists in the
        `DatabaseServiceProvider` enum. If the client type is recognized, it creates the
        corresponding AWS client using the `get_boto3_session().client()` method and returns the
        client object.

        Args:
            - client_type (str): The type of AWS client to generate. This should be one of
            the values defined in the `DatabaseServiceProvider` enum.

        Returns
        -------
            boto3.client: The generated AWS client object.

        Raises: UnRecognizedAWSClientException: If the client type is not recognized,
        an exception is raised with a message indicating the unrecognized client type and the
        available options.

        """
        client = None
        _LOGGER.debug(
            "Generating AWS client for client_type: %s and configurations:%s ",
            client_type,
            str(self.configurations),
        )
        if client_type in DatabaseServiceProvider:
            if client_type == DatabaseServiceProvider.RDS:
                self.rds_client = self.get_boto3_session().client(
                    "rds", region_name=self.configurations.aws_region
                )
                client = self.rds_client
            if client_type == DatabaseServiceProvider.ATHENA:
                self.athena_client = self.get_boto3_session().client(
                    "athena", region_name=self.configurations.aws_region
                )
                client = self.athena_client
        else:
            raise UnRecognizedAWSClientException(
                f"Unrecognized Client Type : {client_type}, "
                f"please choose one from {DatabaseServiceProvider.list()}"
            )

        return client
