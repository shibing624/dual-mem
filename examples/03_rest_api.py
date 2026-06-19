"""Demo 3 — REST API：走真实 HTTP 契约（`/v1/memories/...`）。

用 system1 档演示 REST 端点：add / search（含 messages）/ list / get / delete /
health。本进程内用 Starlette TestClient 拉起真实 ASGI app；线上等价于 `dual-mem serve`。

环境要求：`llm_*` + `embed_*` 两套 API key 都必备；REST 在 `auth_disabled=True` 默认下无需 Bearer。

运行：python examples/03_rest_api.py
期望：所有端点返回 200；search 在 normal 路命中刚写入的事实。
"""
from _common import fresh_storage, section

from fastapi.testclient import TestClient

from dual_mem.api import create_app
from dual_mem.client import MemoryClient
from dual_mem.config import Settings

settings = Settings(mode="system1", storage_dir=fresh_storage("rest"))
client = MemoryClient(settings=settings)
app = create_app(client=client, settings=settings)


def main() -> None:
    with TestClient(app) as http:
        section("GET /health")
        print(" ", http.get("/health").json())

        section("POST /v1/memories/  写入两条（content 形式）")
        for text in ["用户偏好深色主题界面", "用户的母语是中文，第二语言是英语"]:
            r = http.post(
                "/v1/memories/",
                json={"content": text, "app_id": "default", "user_id": "dave"},
            )
            body = r.json()
            print(f"  {r.status_code}  id={body['memory_id'][:8]}  "
                  f"extracted={body.get('extracted_count', 0)}")

        section("POST /v1/memories/  写入一条（messages 形式，多轮对话）")
        r = http.post(
            "/v1/memories/",
            json={
                "messages": [
                    {"role": "user", "content": "顺便提一下，我用 Mac 的快捷键比 Windows 熟练。"},
                    {"role": "assistant", "content": "好的，记下来了。"},
                ],
                "app_id": "default",
                "user_id": "dave",
            },
        )
        print(f"  {r.status_code}  id={r.json()['memory_id'][:8]}")

        section("POST /v1/memories/search  检索语言能力")
        r = http.post(
            "/v1/memories/search",
            json={
                "query": "用户会哪些语言？",
                "app_ids": ["default"],
                "user_id": "dave",
                "min_score": 0.0,
            },
        )
        body = r.json()
        print("  status:", r.status_code, "request_id:", body["request_id"][:8])
        for m in body["memories"]["normal"]:
            print(f"    - ({m['category']} score={m['score']}) {m['content']}")

        section("GET /v1/memories/  列表")
        r = http.get("/v1/memories/", params={"app_id": "default", "user_id": "dave"})
        items = r.json()
        print("  count:", len(items))
        first_id = items[0]["memory_id"]

        section(f"GET /v1/memories/{{id}}  单条")
        print(" ", http.get(f"/v1/memories/{first_id}").json()["content"])

        section(f"PUT /v1/memories/{{id}}  更新")
        print(" ", http.put(
            f"/v1/memories/{first_id}",
            json={"content": "用户偏好浅色主题界面"},
        ).json())

        section("GET /v1/scopes/  列出租户 scope")
        print(" ", http.get("/v1/scopes/", params={"app_id": "default"}).json())

        section("GET /v1/capabilities  工具清单（npm MCP codegen）")
        caps = http.get("/v1/capabilities").json()
        print("  tools:", [t["name"] for t in caps["tools"]])

        section(f"DELETE /v1/memories/{{id}}")
        print(" ", http.delete(f"/v1/memories/{first_id}").json())


if __name__ == "__main__":
    main()
