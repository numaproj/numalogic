import logging

from numalogic.tools.exceptions import ConfigNotFoundError
from omegaconf import OmegaConf
from numalogic.connectors.utils.aws.config import RDSConnectionConfig

_LOGGER = logging.getLogger(__name__)


def load_db_conf(*paths: str) -> RDSConnectionConfig:
    """
    Load database configuration from one or more YAML files.

    Args:
        - paths (str): One or more paths to YAML files containing the database configuration.

    Returns
    -------
    - RDSConfig: An instance of the RDSConfig class representing the loaded database configuration.

    Raises
    ------
    - ConfigNotFoundError: If none of the given configuration file paths exist.

    Example:
        load_db_conf("/path/to/config.yaml", "/path/to/another/config.yaml")
    """
    confs = []
    for _path in paths:
        try:
            conf = OmegaConf.load(_path)
        except FileNotFoundError:
            _LOGGER.warning("Config file path: %s not found. Skipping...", _path)
            continue
        confs.append(conf)

    if not confs:
        _err_msg = f"None of the given conf paths exist: {paths}"
        raise ConfigNotFoundError(_err_msg)

    schema = OmegaConf.structured(RDSConnectionConfig)
    conf = OmegaConf.merge(schema, *confs)
    return OmegaConf.to_object(conf)
