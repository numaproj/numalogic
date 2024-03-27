import structlog
from structlog import processors, stdlib


def configure_logger():
    """Configure struct logger for the UDFs."""
    structlog.configure(
        processors=[
            stdlib.filter_by_level,
            stdlib.add_log_level,
            stdlib.PositionalArgumentsFormatter(),
            processors.TimeStamper(fmt="iso"),
            processors.StackInfoRenderer(),
            processors.format_exc_info,
            processors.UnicodeDecoder(),
            processors.KeyValueRenderer(key_order=["uuid", "event"]),
            stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=stdlib.LoggerFactory(),
        wrapper_class=stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.getLogger(__name__)
