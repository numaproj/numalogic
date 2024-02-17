import logging

import pytest

from numalogic.udfs import set_logger


@pytest.fixture()
def init_logger():
    set_logger()


def some_log(level: str) -> None:
    logger = logging.getLogger()
    getattr(logger, level)("%s log", level)


def test_logging(init_logger, caplog):
    caplog.set_level(logging.DEBUG)
    print()
    some_log("debug")
    some_log("info")
    some_log("warning")
    some_log("error")
    some_log("critical")

    assert caplog.record_tuples == [
        ("root", logging.DEBUG, "debug log"),
        ("root", logging.INFO, "info log"),
        ("root", logging.WARNING, "warning log"),
        ("root", logging.ERROR, "error log"),
        ("root", logging.CRITICAL, "critical log"),
    ]
