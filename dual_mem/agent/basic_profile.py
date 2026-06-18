"""L0_BASIC_INFO 工具：update_basic_user_profile。

把用户稳定的结构化属性（name/age/location/occupation/employer）合并写入 L0 演化链。
每次写入只记录与历史全量 KV 的 diff，diff 存入节点 custom["basic_info_kv"]，
content 渲染为自然语言。旧 head 标 SUPERSEDED 并接入链。
"""

from dual_mem.isolation import build_filter
from dual_mem.providers.embedding import EmbedService
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

TOOL_NAME = "update_basic_user_profile"

TOOL_DESCRIPTION = (
    "Record or update the user's stable structured personal attributes "
    "(name, age, location, occupation, employer). "
    "Call this ONLY when the conversation clearly states or updates one or more of these attributes. "
    "Pass ONLY the attributes that appear in the conversation — omit fields that are not mentioned. "
    "Do NOT repeat these attributes as identity memories; the memory system handles them separately."
)

BASIC_FIELDS = ["name", "age", "location", "occupation", "employer"]

TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "User's full or preferred name."},
        "age": {"type": "integer", "description": "User's age in years."},
        "location": {"type": "string", "description": "User's primary city / region of residence."},
        "occupation": {"type": "string", "description": "User's job title or role."},
        "employer": {"type": "string", "description": "User's employer / company name."},
    },
    "required": [],
    "additionalProperties": False,
}


def openai_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": TOOL_PARAMETERS,
        },
    }


def render_content(kv: dict) -> str:
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


class BasicProfileTool:
    def __init__(self, *, vector: VectorStore, embed: EmbedService):
        self.vector = vector
        self.embed = embed

    def apply(
        self,
        *,
        arguments: dict,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> str | None:
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
        new_node.embedding = self.embed.embed(content)
        self.vector.upsert([new_node])

        if head:
            old = self.vector.get(head.node_id)
            old.is_latest = False
            if new_node.node_id not in old.superseded_by:
                old.superseded_by.append(new_node.node_id)
            old.status = MemoryStatus.SUPERSEDED
            self.vector.upsert([old])

        return new_node.node_id
