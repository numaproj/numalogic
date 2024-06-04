from connectors import (
    DatabaseTypes,
    AWSConfig,
    SSLConfig,
    RDBMSConfig,
    RDSConnectionConfig,
)


def test_aws_config():
    config = AWSConfig(
        aws_assume_role_arn="arn:aws:iam::123456789012:role/roleName",
        aws_assume_role_session_name="Session",
    )
    assert config.aws_assume_role_arn == "arn:aws:iam::123456789012:role/roleName"
    assert config.aws_assume_role_session_name == "Session"


def test_ssl_config():
    ssl = SSLConfig(ca="path_to_ca")
    assert ssl.ca == "path_to_ca"


def test_rdbms_config():
    rdbms = RDBMSConfig(
        endpoint="localhost",
        port=3306,
        database_name="testdb",
        database_username="user",
        database_password="password",
        database_connection_timeout=300,
        database_type=DatabaseTypes.MYSQL,
        ssl_enabled=True,
        ssl=SSLConfig(ca="path_to_ca"),
    )
    assert rdbms.endpoint == "localhost"
    assert rdbms.database_name == "testdb"
    assert rdbms.ssl.ca == "path_to_ca"


def test_rds_config():
    rds = RDSConnectionConfig(
        aws_assume_role_arn="arn:aws:iam::123456789012:role/roleName",
        aws_assume_role_session_name="Session",
        aws_region="us-west-2",
        aws_rds_use_iam=True,
        endpoint="localhost",
        port=3306,
        database_name="testdb",
        database_username="user",
        database_password="password",
        database_connection_timeout=300,
        database_type=DatabaseTypes.MYSQL,
        ssl_enabled=True,
        ssl=SSLConfig(ca="path_to_ca"),
    )
    assert rds.aws_assume_role_arn == "arn:aws:iam::123456789012:role/roleName"
    assert rds.aws_region == "us-west-2"
    assert rds.endpoint == "localhost"
