# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Package entry point for the dual-system layered-memory SDK; exposes the
public MemoryClient facade and the package version.
"""
from dual_mem.client import MemoryClient

__version__ = "0.1.0"

__all__ = ["__version__", "MemoryClient"]
