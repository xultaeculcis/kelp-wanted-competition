from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, TypeVar, Union

from typing_extensions import ParamSpec

from kelp import consts

T = TypeVar("T")
P = ParamSpec("P")


def get_logger(name: str, log_level: Union[int, str] = logging.INFO) -> logging.Logger:
    """
    Builds a `Logger` instance with provided name and log level.

    Args:
        name: The name for the logger.
        log_level: The default log level.

    Returns:
        The logger.

    """

    logger = logging.getLogger(name=name)
    logger.setLevel(log_level)

    # Prevent log messages from propagating to the parent logger
    logger.propagate = False

    # Check if handlers are already set to avoid duplication
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt=consts.logging.FORMAT)
        stream_handler.setFormatter(fmt=formatter)
        logger.addHandler(stream_handler)

    return logger


_timed_logger = get_logger("timed", log_level=logging.INFO)


def timed(func: Callable[P, T]) -> Callable[P, T]:
    """
    This decorator prints the execution time for the decorated function.

    Args:
        func: The function to wrap.

    Returns:
        Wrapper around the function.

    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        _timed_logger.info(f"{func.__qualname__} is running...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        _timed_logger.info(f"{func.__qualname__} ran in {(end - start):.4f}s")
        return result

    return wrapper
