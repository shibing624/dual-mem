"""Demo 3 — REST：真实 HTTP 契约（`/v1/memories/...`）。

system1 档走 add / search / list / get / update / delete / health。
update 追加新版本并返回新 memory_id，旧 id 仍可 get，且 is_latest=false。
delete 归档当前版本，不是物理删除；归档后的 id 不能再 update/delete。

本进程用 Starlette TestClient 拉起 ASGI app，等价于 `dual-mem serve`。
环境要求：jev_*、llm_*、embed_*。auth_disabled=True 时不需要 Bearer。

运行：python examples/03_rest_api.py
期望：写入和检索返回 200；update 的 memory_id 与原 id 不同；delete 用新 id。
"""
from _common import example_settings, section

from fastapi.testclient import TestClient

from dual_mem.api import create_app
from dual_mem.client import MemoryClient

settings = example_settings("system1", "rest")
client = MemoryClient(settings=settings)
app = create_app(client=client, settings=settings)


def main() -> None:
    with TestClient(app) as http:
        section("GET /health")
        print(" ", http.get("/health").json())

        section("POST /v1/memories/  两条 content")
        for text in ["用户偏好深色主题界面", "用户的母语是中文，第二语言是英语"]:
            response = http.post(
                "/v1/memories/",
                json={"content": text, "user_id": "dave"},
            )
            body = response.json()
            print(
                f"  {response.status_code}  raw={body['memory_id'][:8]}  "
                f"({body['processing_time_ms'] / 1000:.2f}s)"
            )

        section("POST /v1/memories/  一条 messages")
        response = http.post(
            "/v1/memories/",
            json={
                "messages": [
                    {"role": "user", "content": "顺便提一下，我用 Mac 的快捷键比 Windows 熟练。"},
                    {"role": "assistant", "content": "好的，记下来了。"},
                ],
                "user_id": "dave",
            },
        )
        print(f"  {response.status_code}  raw={response.json()['memory_id'][:8]}")

        section("POST /v1/memories/search  检索语言")
        response = http.post(
            "/v1/memories/search",
            json={
                "query": "用户会哪些语言？",
                "user_id": "dave",
                "min_score": 0.0,
            },
        )
        body = response.json()
        print("  status:", response.status_code, "request_id:", body["request_id"][:8])
        for item in body["memories"]["normal"]:
            print(f"    - ({item['layer']} {item.get('kind') or item['category']}) {item['content']}")

        section("GET /v1/memories/  当前版本")
        response = http.get("/v1/memories/", params={"user_id": "dave"})
        items = response.json()
        print("  count:", len(items))
        first_id = items[0]["memory_id"]

        section("GET /v1/memories/{id}")
        print(" ", http.get(f"/v1/memories/{first_id}").json()["content"])

        section("PUT /v1/memories/{id}  追加新版本，返回新 id")
        updated = http.put(
            f"/v1/memories/{first_id}",
            json={"content": "用户偏好浅色主题界面"},
        ).json()
        new_id = updated["memory_id"]
        print(f"  new={new_id[:8]}  old={first_id[:8]}  changed={new_id != first_id}")
        old = http.get(f"/v1/memories/{first_id}").json()
        print(f"  old is_latest={old['is_latest']}")

        section("GET /v1/scopes/")
        print(" ", http.get("/v1/scopes/").json())

        section("GET /v1/capabilities")
        caps = http.get("/v1/capabilities").json()
        print("  tools:", [tool["name"] for tool in caps["tools"]])

        section("DELETE /v1/memories/{new_id}  归档当前版本")
        print(" ", http.delete(f"/v1/memories/{new_id}").json())


if __name__ == "__main__":
    main()
