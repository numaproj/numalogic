import time
import logging
from datetime import datetime, timedelta, timezone

import boto3

_LOGGER = logging.getLogger(__name__)


class STSClientManager:
    def __init__(self):
        self.sts_client = boto3.client("sts")
        self.credentials = None

    def assume_role(self, role_arn, role_session_name, duration_seconds=3600):
        """
        Assumes the specified role and retrieves the temporary security credentials.

        Parameters:
        - role_arn (str): The Amazon Resource Name (ARN) of the role to assume.
        - role_session_name (str): A name for the assumed role session.
        - duration_seconds (int): The duration, in seconds, for which the temporary credentials are valid. Default is 3600 seconds (1 hour).

        Returns:
        - None

        Raises:
        - botocore.exceptions.ClientError: If the assume role operation fails.

        """
        response = self.sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=role_session_name,
            DurationSeconds=duration_seconds,
        )
        self.credentials = response["Credentials"]

    def is_token_about_to_expire(self):
        """
        Checks if the token is about to expire.

        Returns:
            bool: True if the token is about to expire within the next 15 minutes, False otherwise.

        """
        if self.credentials:
            if self.credentials["Expiration"] > datetime.now(timezone.utc):
                expiration = self.credentials["Expiration"]
                remaining_time = expiration - datetime.now(timezone.utc)
                return (remaining_time.total_seconds()) <= 15 * 60
        return True

    def get_credentials(self, role_arn, role_session_name):
        """
        Retrieves the AWS IAM credentials for the specified role and role session name.

        Parameters:
        - role_arn (str): The Amazon Resource Name (ARN) of the role to assume.
        - role_session_name (str): A name for the assumed role session.

        Returns:
        - dict: A dictionary containing the temporary security credentials, including the access key, secret key, session token, and expiration time.

        Raises:
        - botocore.exceptions.ClientError: If the assume role operation fails.

        """
        if not self.credentials or self.is_token_about_to_expire():
            _LOGGER.info(
                "Renewing AWS IAM Credentials as existing credentials are expired or does not exists"
            )
            self.assume_role(role_arn, role_session_name)
        else:

            _LOGGER.debug("Using Existing Credentials")
        return self.credentials


# if __name__ == "__main__":
#     sts_client_manager = STSClientManager()
#     credentials = sts_client_manager.get_credentials(
#         "arn:aws:iam::907103919737:role/RDS_IAM_ODL_USER_ROLE",
#         "ml_feature_store_session",
#     )
#     print(credentials)
#     credentials["Expiration"] = datetime.now(timezone.utc) + timedelta(minutes=30)
#     credentials = sts_client_manager.get_credentials(
#         "arn:aws:iam::907103919737:role/RDS_IAM_ODL_USER_ROLE",
#         "ml_feature_store_session",
#     )
#     print(credentials)
