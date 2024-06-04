from connectors import BaseEnum
import logging

logging.basicConfig(level=logging.DEBUG)


class test_aws_init:
    def test_invalid_value_returns_false(self):
        # Arrange
        class MyEnum(BaseEnum):
            VALUE1 = 1
            VALUE2 = 2
            VALUE3 = 3

        # Act
        result = "INVALID" in MyEnum

        # Assert
        assert result is False

    def test_invalid_value_returns_true(self):
        # Arrange
        class MyEnum(BaseEnum):
            VALUE1 = 1
            VALUE2 = 2
            VALUE3 = 3

        # Act
        result = 1 in MyEnum

        # Assert
        assert result is True

    def test_list_method_returns_list_of_values(self):
        # Arrange
        class MyEnum(BaseEnum):
            VALUE1 = 1
            VALUE2 = 2
            VALUE3 = 3

        # Act
        result = MyEnum.list()

        # Assert
        assert result == [1, 2, 3]
