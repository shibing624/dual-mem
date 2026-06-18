# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: System2 cognitive-processing agent: prepares clustered fact materials, then emits
schema/intention ops (single LLM call by default, or a bounded multi-turn loop when
system2_agent_loop is set) executed by the ToolExecutor.
"""
import json

from dual_mem.isolation import build_filter
from dual_mem.providers.llm import is_chinese
from dual_mem.registry import ComponentFactory
from dual_mem.system2.clustering import cluster_facts
from dual_mem.system2.tool_executor import ToolExecutor
from dual_mem.types import Layer, MemoryStatus

SYSTEM2_OPS_PROMPT_ZH = """你是一个认知加工 Agent，从用户的事实聚类中演化高层认知结构（L6 Schema / L7 Intention），并以一组操作（ops）的形式输出。

## L6 Schema 是什么？
Schema 捕获用户在**特定领域**内的**一个**行为模式，含三要素：
- 场景（Circumstance）：该模式发生的领域/话题/场景，不要跨域泛化。
- 模式（Pattern）：用户在该场景下的惯常行为或行动倾向。
- 洞察（Insight）：底层心理驱动力或心智模型。
内容用一句话组合三要素："当[场景]时，用户[模式]——反映了[洞察]。"
规则：一个 Schema 只含一个模式（原子化）；已有 Schema 覆盖该模式 → 用 add_evidence 追加证据，不要重建。
✅ "当做饭时，用户严格按菜谱步骤精确称量——反映了用外部结构管理不确定性的需要。"
❌ "用户对很多事充满热情且追求品质。"（无场景、太泛）

## L7 Intention 是什么？
用户表达的**具体未来事件或计划**，必须有明确行动 + 时间边界（明确或隐含）。
✅ "用户正在准备一场工作面试。"
❌ "用户想成为更好的人。"（无具体行动）

## 输出格式
只输出一个 JSON 对象，键 `ops` 是操作数组，不要任何解释或代码块外文字。每个 op 是以下四类之一：
{{"ops": [
  {{"op": "create_schema", "content": "当...时，用户...——反映...", "tags": ["..."], "evidence": ["fact_id", ...]}},
  {{"op": "create_intention", "content": "...", "tags": ["..."], "evidence": ["fact_id", ...]}},
  {{"op": "add_evidence", "schema_id": "已有schema_id", "evidence": ["fact_id", ...]}},
  {{"op": "add_edge", "from_id": "...", "to_id": "...", "rel": "RELATED_TO"}}
]}}
打标签时优先复用已有标签列表中的标签。证据 fact_id 必须来自聚类中给出的 id。
若数据不足以得出可靠结论，输出 {{"ops": []}}。"""

SYSTEM2_OPS_PROMPT_EN = """You are a cognitive processing Agent. Evolve higher-order cognitive structures (L6 Schema / L7 Intention) from the user's fact clusters, and output them as a list of operations (ops).

## What is an L6 Schema?
A Schema captures ONE behavioral pattern in a SPECIFIC domain, with three components:
- Circumstance: the domain/topic where the pattern is observed; do NOT generalize across domains.
- Pattern: the user's habitual behavior or action tendency in this circumstance.
- Insight: the underlying psychological driver or mental model.
Content is one sentence combining all three: "When [circumstance], the user [pattern] — reflecting [insight]."
Rules: one pattern per Schema (atomic); if an existing Schema already covers the pattern, use add_evidence instead of recreating.
GOOD: "When cooking, the user strictly follows recipe steps and precisely measures ingredients — reflecting a need for external structure to manage uncertainty."
BAD: "The user is passionate about many things and values quality." (no circumstance, too vague)

## What is an L7 Intention?
A CONCRETE future event or plan the user expressed, requiring a clear action + temporal boundedness (explicit or implicit).
GOOD: "The user is preparing for a job interview."
BAD: "The user wants to be a better person." (no concrete action)

## Output format
Output ONLY a JSON object whose key `ops` is the operations array, no prose and nothing outside it. Each op is one of four kinds:
{{"ops": [
  {{"op": "create_schema", "content": "When ..., the user ... — reflecting ...", "tags": ["..."], "evidence": ["fact_id", ...]}},
  {{"op": "create_intention", "content": "...", "tags": ["..."], "evidence": ["fact_id", ...]}},
  {{"op": "add_evidence", "schema_id": "existing_schema_id", "evidence": ["fact_id", ...]}},
  {{"op": "add_edge", "from_id": "...", "to_id": "...", "rel": "RELATED_TO"}}
]}}
Prefer reusing tags from the existing tags list. Evidence fact_ids MUST come from the cluster ids provided.
If the data is insufficient for reliable conclusions, output {{"ops": []}}."""

_S2_LAYERS = [Layer.L2_FACT, Layer.L4_IDENTITY]
_GRAPH_TOPK = 8


def prepare_materials(
    *, factory: ComponentFactory, app_id: str, user_id: str, agent_id: str
) -> dict:
    """Cluster fresh facts and gather existing schemas/tags as System2 input materials."""
    where = build_filter(
        app_ids=[app_id],
        user_id=user_id,
        agent_ids=[agent_id],
        layers=_S2_LAYERS,
        statuses=[MemoryStatus.ACTIVE],
    )
    all_facts = factory.vector.get_many(where)

    fresh = []
    for fact in all_facts:
        if fact.s2_evidence_count == 0:
            full = factory.vector.get(fact.node_id)
            if full is not None and full.embedding:
                fresh.append(full)

    clusters = cluster_facts(
        fresh,
        stage1_sim=factory.settings.cluster_stage1_sim,
        stage2_sim=factory.settings.cluster_stage2_sim,
    )

    graph = factory.graph
    graph_forward: list[dict] = []
    existing_tags: list[str] = []
    seen: set[str] = set()
    if graph is not None:
        tag_set: set[str] = set()
        for cluster in clusters:
            schemas = graph.query_by_embedding(
                layer=Layer.L6_SCHEMA.value,
                user_id=user_id,
                app_ids=[app_id],
                embedding=cluster["centroid_embedding"],
                top_k=_GRAPH_TOPK,
            )
            for node in schemas:
                if node.node_id in seen:
                    continue
                seen.add(node.node_id)
                graph_forward.append(
                    {"node_id": node.node_id, "content": node.content, "tags": node.tags}
                )
                tag_set.update(node.tags)
        existing_tags = sorted(tag_set)

    return {
        "clusters": clusters,
        "graph_forward": graph_forward,
        "existing_tags": existing_tags,
        "stats": {
            "total_facts": len(all_facts),
            "fresh_facts": len(fresh),
            "clusters_found": len(clusters),
            "existing_schemas": len(graph_forward),
        },
    }


def _build_user_message(materials: dict) -> str:
    """Render prepared materials (clusters, existing schemas, tags) into the user prompt."""
    parts = ["## 聚类结果 / Cluster results"]
    for i, cluster in enumerate(materials["clusters"]):
        parts.append(f"### Cluster {i}（主题: {cluster['centroid_text']}）")
        for fact in cluster["facts"]:
            parts.append(f"  - [{fact['layer']}] {fact['content']}  (id={fact['node_id']})")

    forward = materials["graph_forward"]
    if forward:
        parts.append("\n## 已有 Schema / Existing schemas")
        for node in forward:
            parts.append(f"  - {node['content']}  (id={node['node_id']})")

    tags = materials["existing_tags"]
    if tags:
        parts.append("\n## 已有标签 / Existing tags")
        parts.append("  " + ", ".join(tags))

    parts.append("\n请输出 ops JSON 对象 / Output the ops JSON object.")
    return "\n".join(parts)


def _extract_ops(raw) -> list:
    """Normalize the LLM reply (JSON-mode object or bare array) into an ops list."""
    if isinstance(raw, dict):
        ops = raw.get("ops")
        return ops if isinstance(ops, list) else []
    return raw if isinstance(raw, list) else []


class System2Agent:
    """Distills fresh facts into L6 schemas / L7 intentions via clustering and one LLM pass."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    def run(self, *, app_id: str, user_id: str, agent_id: str = "") -> dict:
        """Process one app/user: prepare materials, generate ops, execute them, return stats."""
        materials = prepare_materials(
            factory=self.factory, app_id=app_id, user_id=user_id, agent_id=agent_id
        )
        if not materials["clusters"]:
            return {
                "created_schemas": 0,
                "created_intentions": 0,
                "evidence_added": 0,
                "edges_added": 0,
            }

        cluster_text = " ".join(
            f["content"]
            for cluster in materials["clusters"]
            for f in cluster["facts"]
        )
        system = SYSTEM2_OPS_PROMPT_ZH if is_chinese(cluster_text) else SYSTEM2_OPS_PROMPT_EN
        base_user = _build_user_message(materials)

        if self.factory.settings.system2_agent_loop:
            stats = self._run_loop(
                system=system, base_user=base_user, app_id=app_id, user_id=user_id, agent_id=agent_id
            )
        else:
            raw = self.factory.llm.chat_json(system=system, user=base_user)
            stats = ToolExecutor(factory=self.factory).apply(
                ops=_extract_ops(raw), app_id=app_id, user_id=user_id, agent_id=agent_id
            )

        self._mark_clustered_processed(materials["clusters"])
        return stats

    def _run_loop(
        self, *, system: str, base_user: str, app_id: str, user_id: str, agent_id: str
    ) -> dict:
        """Multi-turn ops emission: each round may add more ops or stop, feeding prior ops back."""
        executor = ToolExecutor(factory=self.factory)
        total = {
            "created_schemas": 0,
            "created_intentions": 0,
            "evidence_added": 0,
            "edges_added": 0,
        }
        history_ops: list = []
        for _ in range(max(1, self.factory.settings.system2_loop_max_iters)):
            user = base_user
            if history_ops:
                user = (
                    base_user
                    + "\n\n## 本轮已生成的 ops（避免重复，可补充新结构或停止）\n"
                    + json.dumps(history_ops, ensure_ascii=False)
                )
            ops = _extract_ops(self.factory.llm.chat_json(system=system, user=user))
            if not ops:
                break
            stats = executor.apply(ops=ops, app_id=app_id, user_id=user_id, agent_id=agent_id)
            for key in total:
                total[key] += stats[key]
            history_ops.extend(ops)
        return total

    def _mark_clustered_processed(self, clusters: list[dict]) -> None:
        """Mark all clustered facts as processed so unused ones are not re-consumed next digest."""
        for cluster in clusters:
            for fact in cluster["facts"]:
                node = self.factory.vector.get(fact["node_id"])
                if node is not None and node.s2_evidence_count == 0:
                    node.s2_evidence_count = 1
                    self.factory.vector.upsert([node])
