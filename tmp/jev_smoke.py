"""MemoryClient smoke: real Venus Jev, deterministic text generation and vectors.

Offline: python tmp/jev_smoke.py --offline
Venus: VENUS_TOKEN=... python tmp/jev_smoke.py --output /tmp/jev-results.json
Or add --token-stdin to pass the token through stdin without a credential file.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.providers.jev import JevClient, JevAnswer


class SmokeEmbed:
    async def embed(self, text: str) -> list[float]:
        values = [b / 255 for b in hashlib.sha256(text.encode()).digest()]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class SmokeGenerator:
    """Use explicit sources as candidates; smoke does not benchmark a generator."""

    async def chat_json_for_content(self, *, content: str, **kwargs) -> dict:
        return {
            "facts": [{"content": content, "tags": []}],
            "identity": [],
            "intentions": [],
        }

    async def chat_json(self, *, system: str, user: str, **kwargs) -> dict:
        facts = json.loads(user)["facts"]
        ids = [f["id"] for f in facts]
        return {
            "memories": [
                {
                    "content": "Recorded evidence summary: "
                    + " ".join(f["content"] for f in facts),
                    "kind": "summary",
                    "evidence_ids": ids,
                }
            ],
            "processed_ids": ids,
        }


class OfflineJev:
    async def evaluate(self, *, state: dict, questions: dict) -> dict:
        result = {}
        for name, question in questions.items():
            criteria = question["criteria"]
            if name.endswith(":action"):
                current = state.get("current", [])
                exact = any(
                    n["content"] == state["proposals"][0]["content"] for n in current
                )
                selected = (
                    "SKIP"
                    if exact
                    else ("ADD_OVERRIDE" if current else "ADD_SUPPLEMENT")
                )
            elif name.endswith(":target"):
                selected = state["current"][0]["id"] if state.get("current") else "NONE"
            else:
                selected = next(
                    (k for k in ("VALID", "PROFILE", "RAW_ONLY") if k in criteria),
                    next(iter(criteria)),
                )
            result[name] = JevAnswer(
                type="choice",
                choice=selected,
                confidence=1,
                probabilities={selected: 1},
            )
        return result


async def run(args: argparse.Namespace) -> dict:
    token = (
        sys.stdin.readline().strip()
        if args.token_stdin
        else os.environ.get("VENUS_TOKEN", "")
    )
    if not args.offline and not token:
        raise RuntimeError("set VENUS_TOKEN or --token-stdin")
    jev = (
        OfflineJev()
        if args.offline
        else JevClient(
            endpoint="http://v2.open.venus.oa.com/llmproxy/v1/systemone",
            api_key=token,
            concurrency=4,
        )
    )
    with tempfile.TemporaryDirectory(prefix="dual-mem-jev-smoke-") as directory:
        c = MemoryClient(
            settings=Settings(mode="dual", storage_dir=directory, embed_dim=32),
            embed=SmokeEmbed(),
            llm=SmokeGenerator(),
            jev=jev,
        )
        phases = []
        for index, source in enumerate(
            [
                "用户目前住在北京。",
                "用户已从北京搬到上海，目前住在上海。",
                "用户又搬回北京，目前住在北京。",
                "用户又搬回北京，目前住在北京。",
            ]
        ):
            started = time.perf_counter()
            result = await c.add(
                content=source,
                user_id="evolution",
                request_id=f"evolution-{index}",
                memory_at=1792022400 + index * 86400,
            )
            nodes = [c.factory.vector.get(nid) for nid in result.extra_node_ids]
            logs = c.factory.cache.list_pipeline_logs(result.request_id)
            decisions = [
                op
                for log in logs
                if log["stage"] == "JEV_DECISION"
                for op in log["payload"]["ops"]
            ]
            phases.append(
                {
                    "source": source,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "extracted_count": result.extracted_count,
                    "decisions": [
                        {
                            "op": op["op"],
                            "update_type": op["update_type"],
                            "choice": op["decision"]["action"]["choice"],
                        }
                        for op in decisions
                    ],
                    "new_ids": [n.node_id for n in nodes],
                }
            )
        started = time.perf_counter()
        await asyncio.gather(
            *(
                c.add(
                    content=f"用户姓名为测试者{i}，常住成都。",
                    user_id=f"parallel-{i}",
                    request_id=f"p{i}",
                )
                for i in range(4)
            )
        )
        concurrent_ms = round((time.perf_counter() - started) * 1000, 1)
        history = c.factory.vector.get_many(
            {"user_id": "evolution", "layer": {"$in": ["L1_FACT", "L2_PROFILE"]}},
            limit=100,
        )
        # Expected memory semantics are assertions, not inferred from successful HTTP.
        assertions = {
            "history_has_three_versions": len(history) == 3,
            "one_latest_head": sum(n.is_latest for n in history) == 1,
            "return_to_old_value_is_override": any(
                d["update_type"] == "OVERRIDE" for d in phases[2]["decisions"]
            ),
            "same_current_value_is_skip": phases[3]["extracted_count"] == 0
            and any(d["choice"] == "SKIP" for d in phases[3]["decisions"]),
        }
        digest = await c.digest()
        summary = {
            "mode": "offline" if args.offline else "venus-real-jev",
            "generation": "deterministic smoke adapter, not a real generative model",
            "model": "jev-1.13.0",
            "phases": phases,
            "four_parallel_writes_ms": concurrent_ms,
            "jev_http_requests": getattr(jev, "request_count", None),
            "digest_processed": digest.processed,
            "assertions": assertions,
            "passed": all(assertions.values()),
        }
        await c.aclose()
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    sys.exit(0 if result["passed"] else 1)
