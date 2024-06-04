from importlib.util import find_spec

from nlconnectors.prometheus import PrometheusFetcher

__all__ = [
    "PrometheusFetcher",
]

if find_spec("boto3"):
    from nlconnectors.rds import RDSFetcher  # noqa: F401

    __all__.append("RDSFetcher")

if find_spec("pydruid"):
    from nlconnectors.druid import DruidFetcher  # noqa: F401

    __all__.append("DruidFetcher")
