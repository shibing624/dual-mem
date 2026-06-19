# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Package entry point for the dual-system layered-memory SDK; exposes the
public MemoryClient facade, the package version, and a one-shot ``enable_logging`` helper.

Logging policy (SDK best practice):
  - All modules use ``logging.getLogger("dual_mem.<sub>")`` — never the root logger.
  - The package attaches a NullHandler at import time so importing the SDK NEVER produces
    "No handlers could be found" warnings nor pollutes the host application's stderr.
  - Application/CLI code that wants to see SDK logs calls ``dual_mem.enable_logging()``.
"""
import logging

from dual_mem.client import MemoryClient

__version__ = "0.1.1"

# Library convention: install a NullHandler on the package logger so the SDK is silent
# by default. The host application is responsible for configuring handlers.
logging.getLogger("dual_mem").addHandler(logging.NullHandler())


def enable_logging(level: int | str = "INFO", *, propagate: bool = False) -> logging.Logger:
    """Enable structured stderr logs for the SDK at ``level`` (e.g. ``"DEBUG"`` / 10).

    Idempotent: repeated calls only update the level. Pass ``propagate=True`` to let
    records bubble up to the host application's root logger configuration.
    """
    logger = logging.getLogger("dual_mem")
    logger.setLevel(level)
    logger.propagate = propagate
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
               for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


__all__ = ["__version__", "MemoryClient", "enable_logging"]
