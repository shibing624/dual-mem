# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Reconsolidation Hook — fires asynchronously after each search. Bumps access
counters for recalled nodes (so the system has "use-it-or-lose-it" signal), creates light
RELATED_TO edges between memories that were co-recalled from different routes, and (in
dual mode) enqueues a reconsolidation task back into System2 for async drain.
"""
import logging

from dual_mem.registry import ComponentFactory

logger = logging.getLogger("dual_mem.retrieval.reconsolidation")

_MAX_BUMP = 20
_MAX_NEW_EDGES = 5


class ReconsolidationHook:
    """Asynchronous read-side hook updating access stats and weak association edges."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    async def process(
        self,
        *,
        query: str,
        recalled_by_route: dict[str, list[str]],
        user_id: str,
        app_id: str,
        agent_id: str = "",
    ) -> dict:
        """Process recalled node ids grouped by route; return summary stats.

        recalled_by_route maps route name (profile/proactive/normal) -> list of node ids.
        """
        all_ids: list[str] = []
        seen: set[str] = set()
        for ids in recalled_by_route.values():
            for nid in ids:
                if nid in seen:
                    continue
                seen.add(nid)
                all_ids.append(nid)

        if not all_ids:
            return {"bumped": 0, "edges_created": 0, "s2_enqueued": False}

        bumped = self._bump_access(all_ids[:_MAX_BUMP])
        edges_created = self._build_co_recall_edges(recalled_by_route)
        s2_enqueued = self._maybe_enqueue_s2(
            query=query,
            node_ids=all_ids,
            user_id=user_id,
            app_id=app_id,
            agent_id=agent_id,
        )
        return {
            "bumped": bumped,
            "edges_created": edges_created,
            "s2_enqueued": s2_enqueued,
        }

    def _bump_access(self, node_ids: list[str]) -> int:
        """Increment access counters for the recalled node ids; tolerate cache failures."""
        try:
            self.factory.cache.bump_access(node_ids)
            return len(node_ids)
        except Exception as exc:
            logger.debug("[reconsolidation] bump_access failed: %s", exc)
            return 0

    def _build_co_recall_edges(self, recalled_by_route: dict[str, list[str]]) -> int:
        """Add RELATED_TO edges between top hits from different routes (dual only)."""
        graph = self.factory.graph
        if graph is None:
            return 0

        routes = [(name, ids) for name, ids in recalled_by_route.items() if ids]
        if len(routes) < 2:
            return 0

        edges_created = 0
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                for src in routes[i][1][:3]:
                    for dst in routes[j][1][:3]:
                        if src == dst or edges_created >= _MAX_NEW_EDGES:
                            continue
                        try:
                            graph.add_edge(from_id=src, to_id=dst, rel="RELATED_TO")
                            edges_created += 1
                        except Exception as exc:
                            logger.debug("[reconsolidation] add_edge failed: %s", exc)
        return edges_created

    def _maybe_enqueue_s2(
        self,
        *,
        query: str,
        node_ids: list[str],
        user_id: str,
        app_id: str,
        agent_id: str,
    ) -> bool:
        """Enqueue a System2 reconsolidation task in dual mode; no-op otherwise."""
        if not self.factory.settings.enable_graph:
            return False
        try:
            self.factory.cache.enqueue_s2_task(
                user_id=user_id,
                app_id=app_id,
                agent_id=agent_id,
                task_type="reconsolidation",
                payload={"query": query, "node_ids": node_ids[:_MAX_BUMP]},
            )
            return True
        except Exception as exc:
            logger.debug("[reconsolidation] enqueue_s2_task failed: %s", exc)
            return False
