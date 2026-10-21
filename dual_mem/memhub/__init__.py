# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MemHub — one local Core and one personal memory shared by every CLI.

- ``config``: Core attachment, Core-owned store paths and local scan cursors.
- ``client``: bearer HTTP client used by CLI hosts and MCP.
- ``sources``: read Agentica / Claude Code / Codex / Cursor / Hermes transcripts
  into turns and submit them incrementally.
- ``hosts``: install the MCP server, recall skill and capture hooks into each CLI.
"""
from dual_mem.memhub.client import HubClient
from dual_mem.memhub.config import (
    CursorStore,
    HubConfig,
    hub_root,
    hub_store_dir,
    hub_upload_dir,
    load_hub_config,
    save_hub_config,
)
from dual_mem.memhub.hosts import HOSTS, hosts_status, install_host, uninstall_host
from dual_mem.memhub.sources import (
    SOURCES,
    build_turns,
    default_source_roots,
    locate_transcript,
    scan_file,
    scan_sources,
)

__all__ = [
    "HOSTS",
    "SOURCES",
    "CursorStore",
    "HubClient",
    "HubConfig",
    "build_turns",
    "default_source_roots",
    "hosts_status",
    "hub_root",
    "hub_store_dir",
    "hub_upload_dir",
    "install_host",
    "load_hub_config",
    "locate_transcript",
    "save_hub_config",
    "scan_file",
    "scan_sources",
    "uninstall_host",
]
