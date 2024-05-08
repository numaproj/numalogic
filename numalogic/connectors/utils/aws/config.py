from dataclasses import dataclass, field
from typing import Optional

from numalogic.connectors.utils.enum import BaseEnum


class DatabaseServiceProvider(str, BaseEnum):
    """
    A class representing the database service providers.

    Attributes
    ----------
        rds (str): Represents the RDS (Relational Database Service) provider.
        athena (str): Represents the Athena provider.

    """

    RDS = "rds"
    ATHENA = "athena"


class DatabaseTypes(str, BaseEnum):
    """
    A class representing different types of databases.

    Attributes
    ----------
        mysql (str): Represents the MySQL database type.
        athena (str): Represents the Athena database type.
    """

    MYSQL = "mysql"
    ATHENA = "athena"


@dataclass
class AWSConfig:
    """
    Class representing AWS configuration.

    Attributes
    ----------
        aws_assume_role_arn (str): The ARN of the IAM role to assume.
        aws_assume_role_session_name (str): The name of the session when assuming the IAM role.
    """

    aws_assume_role_arn: str = ""
    aws_assume_role_session_name: str = ""


@dataclass
class SSLConfig:
    """
    SSLConfig class represents the configuration for SSL/TLS settings.

    Attributes: ca (str): The path to the Certificate Authority (CA) file. Defaults to
    an empty string.

    """

    ca: str = ""


@dataclass
class RDBMSConfig:
    """
    RDBMSConfig class represents the configuration for a Relational Database Management System (
    RDBMS).

    Attributes
    ----------
    endpoint (str): The endpoint or hostname of the database. Defaults to an empty
    string.
    port (int): The port number of the database. Defaults to 3306.
    database_name (str): The name of the database. Defaults to an empty string.
    database_username (str): The username for the database connection. Defaults to an empty string.
    database_password (str): The password for the database connection. Defaults to an empty string.
    database_connection_timeout (int): The timeout duration for the database connection in
    seconds. Defaults to 10.
    database_type (str): The type of the database. Defaults to 'mysql'.
    database_provider (str): The provider of the database service. Defaults to 'rds'.
    ssl_enabled (bool): Flag indicating whether SSL/TLS is enabled for the database connection.
    Defaults to False.
    ssl (Optional[SSLConfig]): The SSL/TLS configuration for the database
    connection. Defaults to an empty SSLConfig object.

    """

    endpoint: str = ""
    port: int = 3306
    database_name: str = ""
    database_username: str = ""
    database_password: str = ""
    database_connection_timeout: int = 10
    database_type: str = DatabaseTypes.MYSQL
    ssl_enabled: bool = False
    ssl: Optional[SSLConfig] = field(default_factory=lambda: SSLConfig())


@dataclass
class RDSConnectionConfig(AWSConfig, RDBMSConfig):
    """
    Class representing the configuration for an RDS (Relational Database Service) instance.

    Inherits from: - AWSConfig: Class representing AWS configuration. - RDBMSConfig: Class
    representing the configuration for a Relational Database Management System (RDBMS).

    Attributes
    ----------
    aws_assume_role_arn (str): The ARN of the IAM role to assume.
    aws_assume_role_session_name (str): The name of the session when assuming the IAM role.
    endpoint (str): The endpoint or hostname of the database. Defaults to an empty string.
    port (int): The port number of the database. Defaults to 3306.
    database_name (str): The name of the database. Defaults to an empty string.
    database_username (str): The username for the database connection. Defaults to an empty string.
    database_password (str): The password for the database connection. Defaults to an empty string.
    database_connection_timeout (int): The timeout duration for the database connection in seconds.
     Defaults to 10.
    database_type (str): The type of the database. Defaults to 'mysql'.
    database_provider (str): The provider of the database service. Defaults to 'rds'.
    ssl_enabled (bool): Flag indicating whether SSL/TLS is enabled for the database connection.
    Defaults to False.
    ssl (Optional[SSLConfig]): The SSL/TLS configuration for the database connection.
    Defaults to an empty SSLConfig object.
    aws_region (str): The AWS region for the RDS instance.
    aws_rds_use_iam (bool): Flag indicating whether to use IAM authentication for the RDS instance.
     Defaults to False.

    """

    aws_region: str = ""
    aws_rds_use_iam: bool = False
    database_provider: str = DatabaseServiceProvider.RDS
