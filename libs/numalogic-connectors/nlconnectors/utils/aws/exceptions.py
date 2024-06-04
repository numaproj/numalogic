class AWSException(Exception):
    """Custom exception class for grouping all AWS Exceptions together."""

    pass


class UnRecognizedAWSClientException(AWSException):
    """
    Custom exception class for handling unrecognized AWS clients.

    This exception is raised when an unrecognized AWS client is requested from the
    Boto3ClientManager class.

    Usage: This exception can be raised when attempting to retrieve an AWS client that is not
    recognized by the Boto3ClientManager.

    """

    pass


class UnRecognizedDatabaseTypeException(AWSException):
    """
    Exception raised when an unrecognized database type is encountered.

    This exception is raised when a database type is encountered that is not recognized or
    supported by the application. It serves as a way to handle and communicate errors related to
    unrecognized database types.

    Usage: This exception can be raised when attempting to connect to a database with an
    unrecognized type, or when performing operations on a database with an unrecognized type. It
    can be caught and handled to provide appropriate error messages or take necessary actions.
    """

    pass


class UnRecognizedDatabaseServiceProviderException(Exception):
    """
    Exception raised when an unrecognized database service provider is encountered.

    This exception is raised when a database service provider is not recognized or supported by
    the application. It can be used to handle cases where a specific database service provider
    is required, but the provided one is not supported.


    Usage: raise UnRecognizedDatabaseServiceProviderException("The provided database service
    provider is not recognized or supported.")
    """

    pass
