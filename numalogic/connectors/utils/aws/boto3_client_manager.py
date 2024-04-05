from boto3 import Session
import logging

from numalogic.connectors.rds._config import DatabaseServiceProvider
from numalogic.connectors.utils.aws.exceptions import UnRecognizedAWSClientException
from numalogic.connectors.utils.aws.sts_client_manager import STSClientManager

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


class Boto3ClientManager:

    def __init__(self, configurations):
        """
        Initializes the Boto3ClientManager with the given configurations.

        The Boto3ClientManager is responsible for managing AWS clients for different services like RDS and Athena.
        It uses the configurations to create the clients and manage their sessions.

        Args: configurations (object): An object containing the necessary configurations. The configurations should
        include: - aws_assume_role_arn: The ARN of the role to assume for AWS services. -
        aws_assume_role_session_name: The session name to use when assuming the role. - endpoint: The endpoint for
        the AWS service. - port: The port to use for the AWS service. - database_username: The username for the
        database. - aws_region: The AWS region where the services are located.

        Attributes:
            rds_client (boto3.client): The client for AWS RDS service. Initialized as None.
            athena_client (boto3.client): The client for AWS Athena service. Initialized as None.
            configurations (object): The configurations for the AWS services.
            sts_client_manager (STSClientManager): The STSClientManager for managing AWS STS sessions.

        """
        self.rds_client = None
        self.athena_client = None
        self.configurations = configurations
        self.sts_client_manager = STSClientManager()

    def get_boto3_session(self) -> Session:
        """
        Returns a Boto3 session object with the necessary credentials.

        This method retrieves the credentials from the STSClientManager using the given AWS assume role ARN and
        session name. It then creates a Boto3 session object with the retrieved credentials and returns it.

        Returns:
            Session: A Boto3 session object with the necessary credentials.

        """
        credentials = self.sts_client_manager.get_credentials(
            self.configurations.aws_assume_role_arn,
            self.configurations.aws_assume_role_session_name,
        )
        tmp_access_key = credentials["AccessKeyId"]
        tmp_secret_key = credentials["SecretAccessKey"]
        security_token = credentials["SessionToken"]
        boto3_session = Session(
            aws_access_key_id=tmp_access_key,
            aws_secret_access_key=tmp_secret_key,
            aws_session_token=security_token,
        )
        return boto3_session

    def get_rds_token(self, rds_boto3_client) -> str:
        """
        Generates an RDS authentication token using the provided RDS boto3 client.

        This method generates an RDS authentication token by calling the 'generate_db_auth_token' method of the
        provided RDS boto3 client. The authentication token is generated using the following parameters: -
        DBHostname: The endpoint of the RDS database. - Port: The port number of the RDS database. - DBUsername: The
        username for the RDS database. - Region: The AWS region where the RDS database is located.

        Parameters:
            rds_boto3_client (boto3.client): The RDS boto3 client used to generate the authentication token.

        Returns:
            str: The generated RDS authentication token.

        """
        rds_token = rds_boto3_client.generate_db_auth_token(
            DBHostname=self.configurations.endpoint,
            Port=self.configurations.port,
            DBUsername=self.configurations.database_username,
            Region=self.configurations.aws_region,
        )
        return rds_token

    def get_client(self, client_type: str):
        """
        Generates an AWS client based on the provided client type.

        This method generates an AWS client based on the provided client type. It first checks if the client type is
        recognized by checking if it exists in the `DatabaseServiceProvider` enum. If the client type is recognized,
        it creates the corresponding AWS client using the `get_boto3_session().client()` method and returns the
        client object.

        Parameters: client_type (str): The type of AWS client to generate. This should be one of the values defined
        in the `DatabaseServiceProvider` enum.

        Returns:
            boto3.client: The generated AWS client object.

        Raises: UnRecognizedAWSClientException: If the client type is not recognized, an exception is raised with a
        message indicating the unrecognized client type and the available options.

        """
        _LOGGER.debug(
            f"Generating AWS client for client_type: {client_type} , and configurations: {str(self.configurations)}"
        )
        if client_type in DatabaseServiceProvider:
            if client_type == DatabaseServiceProvider.rds.value:
                self.rds_client = self.get_boto3_session().client(
                    "rds", region_name=self.configurations.aws_region
                )
                return self.rds_client
            if client_type == DatabaseServiceProvider.athena.value:
                self.athena_client = self.get_boto3_session().client(
                    "athena", region_name=self.configurations.aws_region
                )
        else:
            raise UnRecognizedAWSClientException(
                f"Unrecognized Client Type : {client_type}, please choose one from {DatabaseServiceProvider.list()}"
            )
