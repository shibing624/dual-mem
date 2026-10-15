"""Drain legacy deferred tasks using the same Jev System1 pipeline.

New writes reconcile before returning. This worker exists only to consume queues
created by earlier SDK releases; there is no skip-model or heuristic path.
"""

from dual_mem.registry import ComponentFactory
from dual_mem.agent.mem_agent import MemAgent


class ReconcilerWorker:
    def __init__(self, *, factory: ComponentFactory) -> None:
        self.factory = factory

    async def reconcile_pending(
        self, *, app_id: str, user_id: str, agent_id: str = ""
    ) -> int:
        tasks = self.factory.cache.list_pending_reconcile_tasks(
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
        )
        scope = (app_id, user_id, agent_id)
        agent = MemAgent(factory=self.factory)
        for task in tasks:
            fetched = self.factory.vector.get_by_ids(task["node_ids"])
            if set(fetched) != set(task["node_ids"]):
                raise RuntimeError("legacy task has missing evidence; retain pending")
            originals = list(fetched.values())
            if not originals:
                self.factory.cache.mark_reconcile_task_done(task["id"])
                continue
            if any((n.app_id, n.user_id, n.agent_id) != scope for n in originals):
                raise RuntimeError("legacy task crosses tenant scope")
            meta = [
                {
                    "content": n.content,
                    "layer": n.layer.value,
                    "kind": n.kind or n.category.value,
                    "source_ids": [n.node_id],
                }
                for n in originals
            ]
            ops = await agent.reconciler.reconcile(
                new_memories=[n.content for n in originals],
                new_memories_meta=meta,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                current_time="",
                exclude_ids=task["node_ids"],
            )
            nodes, expected, archives = await agent._prepare_nodes(
                ops, originals[0], f"legacy:{task['id']}:{scope}"
            )
            # The legacy extracted representations remain as archived ancestors.
            expected.update({n.node_id: n.revision for n in originals if n.is_latest})
            archives.extend(
                n.node_id
                for n in originals
                if n.is_latest and n.node_id not in archives
            )
            if not self.factory.vector.apply_version_plan(
                nodes=nodes,
                expected=expected,
                archives=archives,
                scope=scope,
                expected_epoch=ops[0].scope_epoch,
            ):
                raise RuntimeError("legacy migration targets changed; retain pending")
            self.factory.cache.enqueue_s2_task(
                user_id=user_id, app_id=app_id, agent_id=agent_id
            )
            self.factory.cache.mark_reconcile_task_done(task["id"])
        return len(tasks)
