"""Demo 4 — REST API：走真实 HTTP 契约（FastAPI）。

用 lite 档（快、零 LLM）演示 REST 端点：add / search / list / get / delete / health。
这里用 Starlette TestClient 在进程内拉起真实 ASGI app；线上等价于 `dual-mem serve`。

运行：python examples/04_rest_api.py
"""
from _common import fresh_storage, section

from fastapi.testclient import TestClient

from dual_mem.api import create_app
from dual_mem.client import MemoryClient
from dual_mem.config import Settings

settings = Settings(mode="lite", storage_dir=fresh_storage("rest"))
client = MemoryClient(settings=settings)
app = create_app(client=client, settings=settings)


def main() -> None:
    with TestClient(app) as http:
        section("GET /health")
        print(" ", http.get("/health").json())

        section("POST /v1/memories/  写入两条")
        for text in ["用户偏好深色主题界面", "用户的母语是中文，第二语言是英语"]:
            r = http.post(
                "/v1/memories/",
                json={"content": text, "app_id": "default", "user_id": "dave"},
            )
            print(" ", r.status_code, r.json()["memory_id"][:8])

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
        print("  status:", r.status_code)
        for m in body["memories"]["normal"]:
            print("    -", m["content"])

        section("GET /v1/memories/  列表")
        r = http.get("/v1/memories/", params={"app_id": "default", "user_id": "dave"})
        items = r.json()
        print("  count:", len(items))
        first_id = items[0]["memory_id"]

        section(f"GET /v1/memories/{{id}}  单条")
        print(" ", http.get(f"/v1/memories/{first_id}").json()["content"])

        section(f"DELETE /v1/memories/{{id}}")
        print(" ", http.delete(f"/v1/memories/{first_id}").json())


if __name__ == "__main__":
    main()
