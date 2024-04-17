class RDSException(Exception):
    """Custom exception class for grouping all AWS Exceptions together."""

    pass


class RDSFetcherConfValidationException(RDSException):
    pass
