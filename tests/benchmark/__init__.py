"""Recall regression harness — opt-in via `pytest -m benchmark`.

Mock-only by default: a tiny in-memory dataset of (write_corpus, query, expected_substr)
exercised against ``MemoryClient`` with the FakeLLM/FakeEmbed fixtures from conftest. The
harness measures recall@k for the hybrid reader. Real-API benchmarks live elsewhere.

Run all benchmarks::
    pytest tests/benchmark/ -m benchmark -v

Skip them in normal runs (default; they are -m benchmark gated)::
    pytest tests/ -q
"""
