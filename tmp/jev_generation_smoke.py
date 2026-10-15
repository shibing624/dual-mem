"""Real generation + Venus Jev smoke; deterministic embedding, semantic assertions.

Set GENERATION_API_KEY, GENERATION_BASE_URL, GENERATION_MODEL, and VENUS_TOKEN.
Credentials are read only from environment/stdin and never included in reports.
This is a four-sample smoke, not a quality or speed benchmark.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_smoke import SmokeEmbed
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.providers.jev import JevClient
from dual_mem.providers.llm import LLMClient


async def run(args: argparse.Namespace) -> dict:
    token = (
        sys.stdin.readline().strip()
        if args.token_stdin
        else os.environ.get("VENUS_TOKEN", "")
    )
    generation_key = os.environ.get("GENERATION_API_KEY", "")
    if not token or not generation_key:
        raise RuntimeError(
            "Provide Venus and generation credentials via environment/stdin"
        )
    generator = LLMClient(
        base_url=os.environ["GENERATION_BASE_URL"],
        api_key=generation_key,
        model=os.environ["GENERATION_MODEL"],
        timeout=60,
    )
    jev = JevClient(
        endpoint="http://v2.open.venus.oa.com/llmproxy/v1/systemone", api_key=token
    )
    writes = []
    with tempfile.TemporaryDirectory() as directory:
        client = MemoryClient(
            settings=Settings(mode="dual", storage_dir=directory, embed_dim=32),
            embed=SmokeEmbed(),
            llm=generator,
            jev=jev,
        )
        try:
            for source in [
                "我目前住在北京。",
                "我从北京搬到上海了，现在住在上海。",
                "我又搬回北京了，目前住在北京。",
                "我目前住在北京。",
            ]:
                before = jev.request_count
                started = time.perf_counter()
                result = await client.add(content=source, user_id="generation-smoke")
                writes.append(
                    {
                        "source": source,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000),
                        "jev_requests": jev.request_count - before,
                        "decisions": [d.to_dict() for d in result.decisions],
                    }
                )
            digest = await client.digest()
            nodes = client.factory.vector.get_many(
                {"user_id": "generation-smoke"}, limit=100
            )
            assertions = {
                "return_is_override": any(
                    d.get("update_type") == "OVERRIDE" for d in writes[2]["decisions"]
                ),
                "repeat_is_skip": bool(writes[3]["decisions"])
                and all(d["action"] == "SKIP" for d in writes[3]["decisions"]),
                "one_latest_profile": sum(
                    n.is_latest and n.layer.value == "L2_PROFILE" for n in nodes
                )
                == 1,
            }
            report = {
                "passed": all(assertions.values()),
                "assertions": assertions,
                "digest_processed": digest.processed,
                "nodes": [
                    {
                        "id": n.node_id,
                        "content": n.content,
                        "layer": n.layer.value,
                        "kind": n.kind,
                        "is_latest": n.is_latest,
                        "supersedes": n.supersedes,
                    }
                    for n in nodes
                ],
            }
        except Exception as exc:
            report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            await client.aclose()
    report.update(
        generator=generator.model,
        jev="jev-1.13.0",
        embedding="deterministic smoke adapter",
        writes=writes,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--output", type=Path)
    result = asyncio.run(run(parser.parse_args()))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if output := parser.parse_args().output:
        output.write_text(rendered + "\n")
    print(rendered)
    sys.exit(0 if result["passed"] else 1)
