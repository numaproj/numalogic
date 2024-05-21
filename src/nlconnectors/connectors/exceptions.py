class ConnectorFetcherException(Exception):
    """Custom exception class for grouping all Connector Exceptions together."""

    pass


class RDSFetcherConfValidationException(ConnectorFetcherException):
    """A custom exception class for handling validation errors in RDSFetcherConf."""

    pass
