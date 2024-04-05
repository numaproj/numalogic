import unittest

from numalogic.connectors.rds._config import DatabaseServiceProvider, DatabaseTypes, AWSConfig, \
    SSLConfig, RDBMSConfig, RDSConfig


class TestDataClasses(unittest.TestCase):

    def test_aws_config(self):
        config = AWSConfig(aws_assume_role_arn='arn:aws:iam::123456789012:role/roleName',
                           aws_assume_role_session_name='Session')
        self.assertEqual(config.aws_assume_role_arn, 'arn:aws:iam::123456789012:role/roleName')
        self.assertEqual(config.aws_assume_role_session_name, 'Session')

    def test_ssl_config(self):
        ssl = SSLConfig(ca="path_to_ca")
        self.assertEqual(ssl.ca, "path_to_ca")

    def test_rdbms_config(self):
        rdbms = RDBMSConfig(endpoint="localhost", port=3306,
                            database_name="testdb", database_username="user",
                            database_password="password", database_connection_timeout=300,
                            database_type=DatabaseTypes.mysql.value, ssl_enabled=True,
                            ssl=SSLConfig(ca="path_to_ca"))
        self.assertEqual(rdbms.endpoint, "localhost")
        self.assertEqual(rdbms.database_name, "testdb")
        self.assertEqual(rdbms.ssl.ca, "path_to_ca")

    def test_rds_config(self):
        rds = RDSConfig(aws_assume_role_arn='arn:aws:iam::123456789012:role/roleName',
                        aws_assume_role_session_name='Session', aws_region="us-west-2",
                        aws_rds_use_iam=True, endpoint="localhost", port=3306,
                        database_name="testdb", database_username="user",
                        database_password="password", database_connection_timeout=300,
                        database_type=DatabaseTypes.mysql.value, ssl_enabled=True,
                        ssl=SSLConfig(ca="path_to_ca"))
        self.assertEqual(rds.aws_assume_role_arn, "arn:aws:iam::123456789012:role/roleName")
        self.assertEqual(rds.aws_region, "us-west-2")
        self.assertEqual(rds.endpoint, "localhost")


if __name__ == '__main__':
    unittest.main()
