# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Lightweight, zero-LLM query understanding for the read path. Combines the
existing intent / time-range / keyword heuristics into a single QueryUnderstanding result
that downstream readers can use to choose target layers and detect temporal queries.
"""
from dataclasses import dataclass, field
from datetime import datetime

from dual_mem.retrieval.intent import (
    classify_intent,
    extract_keywords,
    parse_time_range,
    wants_evolution_history,
)
from dual_mem.types import Layer


# Layer routing per intent: profile-heavy queries lean on L0/L4/L6, conceptual on L4/L6,
# factual on L2/L3, navigational stays narrow on L2 (exact-match style content).
_INTENT_TO_LAYERS: dict[str, list[Layer]] = {
    "FACTUAL": [Layer.L2_FACT, Layer.L3_SUMMARY],
    "CONCEPTUAL": [Layer.L4_IDENTITY, Layer.L6_SCHEMA, Layer.L2_FACT],
    "NAVIGATIONAL": [Layer.L2_FACT, Layer.L1_RAW],
}


@dataclass
class QueryUnderstanding:
    """Heuristic query analysis: intent, time range, target layers, keywords."""

    raw_query: str
    intent: str = "FACTUAL"
    keywords: list[str] = field(default_factory=list)
    has_temporal: bool = False
    time_from: int | None = None
    target_layers: list[Layer] = field(default_factory=list)
    # True when the query asks about a past/changed state → show evolution-chain history.
    wants_evolution: bool = False


def understand(query: str, *, now: datetime | None = None) -> QueryUnderstanding:
    """Run the heuristic analyzers over a raw query and assemble the result."""
    raw = query or ""
    intent = classify_intent(raw)
    keywords = extract_keywords(raw)
    time_from = parse_time_range(raw, now=now)
    target_layers = list(_INTENT_TO_LAYERS.get(intent, [Layer.L2_FACT]))
    return QueryUnderstanding(
        raw_query=raw,
        intent=intent,
        keywords=keywords,
        has_temporal=time_from is not None,
        time_from=time_from,
        target_layers=target_layers,
        wants_evolution=wants_evolution_history(raw),
    )
