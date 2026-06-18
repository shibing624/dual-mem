# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tenant-isolation helpers: build composite isolation keys and translate
scope/layer/status constraints into vector-store metadata filters.
"""
from dual_mem.types import Layer, MemoryStatus


def isolation_key(user_id: str, agent_id: str = "", session_id: str = "") -> str:
    """Build a composite isolation key from user/agent/session identifiers."""
    return f"{user_id}::{agent_id}::{session_id}"


def build_filter(
    *,
    app_ids: list[str],
    user_id: str,
    agent_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    layers: list[Layer] | None = None,
    statuses: list[MemoryStatus] | None = None,
    created_after: int | None = None,
) -> dict:
    """Build a metadata where-filter from scope, layer, status and time constraints."""
    where: dict = {
        "app_id": {"$in": app_ids},
        "user_id": user_id,
    }
    if agent_ids is not None:
        where["agent_id"] = {"$in": agent_ids}
    if session_ids is not None:
        where["session_id"] = {"$in": session_ids}
    if layers is not None:
        where["layer"] = {"$in": [layer.value for layer in layers]}
    if statuses is not None:
        where["status"] = {"$in": [status.value for status in statuses]}
    if created_after is not None:
        where["gmt_created"] = {"$gte": created_after}
    return where
