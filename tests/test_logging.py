"""Unit test to test logging"""

import logging
from panaroma_stitcher.logging import config_logger

logger = logging.getLogger(__name__)


def test_config_logger() -> None:
    """Unit test to test the logger"""
    config_logger(10)
    assert logger.name == "tests.test_logging"
    assert logger.root.level == 10
    assert logger.handlers == []
