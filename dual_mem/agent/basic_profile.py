# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: L0_BASIC_INFO profile updater: merges stable structured user attributes into an
evolution chain, storing only the KV diff and superseding the old head.
"""
import logging
from dataclasses import dataclass

from dual_mem.isolation import build_filter
from dual_mem.providers.embedding import EmbedService
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.agent.basic_profile")

BASIC_FIELDS = ["name", "age", "location", "occupation", "employer"]


def render_content(kv: dict) -> str:
    """Render a basic-info KV dict into a natural-language sentence (empty when no fields)."""
    parts = []
    for key in BASIC_FIELDS:
        value = kv.get(key)
        if value is None:
            continue
        s = str(value).strip()
        if not s:
            continue
        parts.append(f"{key} is {s}")
    if not parts:
        return ""
    return "The user's " + ", ".join(parts) + "."


def _sanitize_arguments(arguments: dict) -> dict:
    """Keep only valid basic fields, trimming strings and coercing age to int."""
    result: dict = {}
    for k in BASIC_FIELDS:
        if k not in arguments:
            continue
        v = arguments[k]
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in ("null", "none"):
                continue
            result[k] = s
        elif isinstance(v, (int, float)) and k == "age":
            result[k] = int(v)
    return result


@dataclass
class PreparedL0:
    """L0 node ready to persist once an embedding vector is available."""

    node: MemoryNode
    head: MemoryNode | None


class BasicProfileTool:
    """Maintains the L0 basic-info evolution chain from extracted profile attributes."""

    def __init__(self, *, vector: VectorStore, embed: EmbedService):
        self.vector = vector
        self.embed = embed

    def prepare(
        self,
        *,
        arguments: dict,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> PreparedL0 | None:
        """Build an L0 head candidate without embedding or upsert (for post-extract batching)."""
        new_kv = _sanitize_arguments(arguments)
        if not new_kv:
            return None

        l0_nodes = self.vector.get_many(
            build_filter(
                app_ids=[app_id],
                user_id=user_id,
                agent_ids=[agent_id],
                layers=[Layer.L0_BASIC_INFO],
            )
        )
        l0_nodes.sort(key=lambda n: n.gmt_created)

        full_kv: dict = {}
        for node in l0_nodes:
            kv = (node.custom or {}).get("basic_info_kv") or {}
            for k, v in kv.items():
                if k in BASIC_FIELDS:
                    full_kv[k] = v

        diff_kv = {}
        for k, v in new_kv.items():
            old_v = full_kv.get(k)
            if old_v is None or str(v) != str(old_v):
                diff_kv[k] = v
        if not diff_kv:
            return None

        head = next((n for n in l0_nodes if n.is_latest), None)
        content = render_content(diff_kv)
        new_node = MemoryNode(
            content=content,
            layer=Layer.L0_BASIC_INFO,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            tags=["basic_info"],
            status=MemoryStatus.ACTIVE,
            is_latest=True,
            supersedes=[head.node_id] if head else [],
            custom={"basic_info_kv": diff_kv},
        )
        return PreparedL0(node=new_node, head=head)

    def commit(self, prepared: PreparedL0, embedding: list[float]) -> str:
        """Persist a prepared L0 node with a precomputed embedding."""
        new_node = prepared.node
        new_node.embedding = embedding
        self.vector.upsert([new_node])

        head = prepared.head
        if head is not None:
            self.vector.mark_superseded(head.node_id, superseded_by_id=new_node.node_id)

        logger.debug(
            "basic_profile committed user=%s diff_keys=%s superseded=%s",
            new_node.user_id,
            list((new_node.custom or {}).get("basic_info_kv", {}).keys()),
            head is not None,
        )
        return new_node.node_id

    async def apply(
        self,
        *,
        arguments: dict,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> str | None:
        """Apply a profile update: write the diff as a new L0 head, supersede the old head."""
        prepared = self.prepare(
            arguments=arguments,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        if prepared is None:
            return None
        embedding = await self.embed.embed(prepared.node.content)
        return self.commit(prepared, embedding)
