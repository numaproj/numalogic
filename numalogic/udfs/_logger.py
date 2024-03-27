import logging
import structlog
from structlog import processors, stdlib


def configure_logger(log_level: str = "INFO"):
    """Configure struct logger for the UDFs."""
    shared_processors = [
        stdlib.add_log_level,
        stdlib.PositionalArgumentsFormatter(),
        processors.TimeStamper(fmt="iso"),
        processors.StackInfoRenderer(),
        processors.format_exc_info,
        processors.UnicodeDecoder(),
    ]
    structlog.configure(
        processors=[
            stdlib.add_log_level,
            stdlib.PositionalArgumentsFormatter(),
            processors.TimeStamper(fmt="iso"),
            processors.StackInfoRenderer(),
            processors.format_exc_info,
            processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            processors.KeyValueRenderer(key_order=["uuid", "event"]),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = structlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    return structlog.getLogger(__name__)
