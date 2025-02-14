"""Logger initialization"""

import logging
import logging.config


def config_logger(loglevel: int) -> None:
    """Initialize a custom logger"""
    default_logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d (%(process)d)] | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": loglevel,
        },
    }
    logging.config.dictConfig(default_logging_config)
