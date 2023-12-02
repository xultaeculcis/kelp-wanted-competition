import argparse
from typing import Type, TypeVar

from kelp.core.configs.base import ConfigBase
from kelp.utils.logging import get_logger

_logger = get_logger(__name__)
T = TypeVar("T", bound=ConfigBase)


def parse_args(parser: argparse.ArgumentParser, cfg_cls: Type[T]) -> T:
    """
    Parses Command Line arguments and returns the script config model.

    Args:
        parser: The instance of `argparse.ArgumentParser` to use.
        cfg_cls: The class of the config to use. Anything inheriting from `ConfigBase` will work.

    Returns:
         A config instance of type given by `cfg_cls`.

    """
    known_args, unknown_args = parser.parse_known_args()
    _logger.info(f"Unknown args: {unknown_args}")
    cfg = cfg_cls(**vars(known_args))
    if cfg.log_config:
        _logger.info(f"Running with following config: {cfg}")

    return cfg
