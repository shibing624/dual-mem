# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Query intent classification (navigational/conceptual/factual), bilingual
keyword extraction and Chinese relative-time parsing into a created_after timestamp.
"""
import re
from datetime import datetime, timedelta

KEYWORD_MAX_COUNT = 10

INTENT_WEIGHTS_2CHANNEL: dict[str, dict[str, float]] = {
    "NAVIGATIONAL": {"vec": 0.3, "bm25": 1.5},
    "FACTUAL": {"vec": 1.2, "bm25": 0.8},
    "CONCEPTUAL": {"vec": 1.0, "bm25": 0.6},
}


_NAV_PATTERNS = [
    r"`[^`]+`",
    r'"[^"]{2,}"',
    r"'[^']{2,}'",
    r"[/\\][\w.\-]+[/\\]",
    r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b",
    r"\b[A-Za-z][a-z0-9]+[A-Z][A-Za-z0-9]*\b",
    r"\bmem-[a-f0-9]{8,}\b",
    r"\b[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\b",
    r"\bv?\d+\.\d+(?:\.\d+)?\b",
    r"\b[a-f0-9]{16,}\b",
    r"https?://\S+",
]

_NAV_REGEX = re.compile("|".join(_NAV_PATTERNS))

_CONCEPTUAL_TRIGGERS = {
    "how", "why", "explain", "approach", "strategy", "tend", "overall",
    "architecture", "design", "philosophy", "pattern", "style",
    "in general", "generally",
    "怎么", "为什么", "如何", "倾向", "风格", "整体", "一般", "通常",
    "总体", "模式",
}

_STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "having",
    "my", "mine", "your", "yours", "our", "ours", "their", "theirs",
    "i", "you", "we", "they", "he", "she", "it", "me", "him", "her", "us", "them",
    "this", "that", "these", "those",
    "what", "when", "where", "who", "whom", "how", "why", "which",
    "and", "or", "but", "not", "no", "nor",
    "for", "to", "of", "in", "on", "at", "by", "from", "with", "about",
    "as", "if", "then", "than", "so", "too", "very",
    "can", "could", "would", "should", "may", "might", "will", "shall",
    "am",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def is_navigational(query: str) -> bool:
    """True if the query contains navigational signals (identifiers, paths, URLs, versions)."""
    return bool(_NAV_REGEX.search(query or ""))


def is_conceptual(query: str) -> bool:
    """True if the query contains conceptual trigger words (how/why/design, 为什么/如何...)."""
    if not query:
        return False
    q_lower = query.lower()
    return any(trig in q_lower for trig in _CONCEPTUAL_TRIGGERS)


def classify_intent(query: str) -> str:
    """Classify intent with priority NAVIGATIONAL > CONCEPTUAL > FACTUAL."""
    if is_navigational(query):
        return "NAVIGATIONAL"
    if is_conceptual(query):
        return "CONCEPTUAL"
    return "FACTUAL"


def extract_keywords(query: str) -> list[str]:
    """Extract bilingual keywords (ascii >=3 non-stopword, CJK >=2), de-duped, order-preserving, <=10."""
    if not query:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for tok in _TOKEN_RE.findall(query):
        is_ascii = tok.isascii()
        t = tok.lower() if is_ascii else tok
        if is_ascii:
            if len(t) < 3 or t in _STOPWORDS_EN:
                continue
        elif len(t) < 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        result.append(t)
        if len(result) >= KEYWORD_MAX_COUNT:
            break
    return result


# History-seeking phrases: questions about how a fact/preference changed over time. These
# benefit from seeing the full evolution chain (superseded versions), not just the current head.
_EVOLUTION_TRIGGERS = (
    "previous", "previously", "used to", "before", "earlier", "originally",
    "no longer", "changed", "switch", "switched", "former", "past",
    "之前", "以前", "原来", "原本", "曾经", "改成", "改为", "不再", "过去", "最初", "当初",
)


def wants_evolution_history(query: str) -> bool:
    """True if the query asks about a past/changed state (show evolution-chain history)."""
    if not query:
        return False
    q = query.lower()
    return any(trig in q for trig in _EVOLUTION_TRIGGERS)


_RECENT_RE = re.compile(r"(?:最近|近|过去)\s*(\d+)\s*天")
_AGO_RE = re.compile(r"(\d+)\s*天前")


def _day_start(dt: datetime) -> datetime:
    """Return the start-of-day (00:00:00) for the given datetime."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_time_range(query: str, now: datetime | None = None) -> int | None:
    """Parse common Chinese relative-time phrases into a created_after unix timestamp, else None."""
    if not query:
        return None
    now = now or datetime.now()
    today = _day_start(now)

    if "前天" in query:
        return int((today - timedelta(days=2)).timestamp())
    if "昨天" in query:
        return int((today - timedelta(days=1)).timestamp())
    if "今天" in query:
        return int(today.timestamp())

    monday = today - timedelta(days=today.weekday())
    if "上周" in query or "上星期" in query or "上个星期" in query:
        return int((monday - timedelta(days=7)).timestamp())
    if "这周" in query or "本周" in query or "这个星期" in query:
        return int(monday.timestamp())

    if "上个月" in query or "上月" in query:
        last_month_end = today.replace(day=1) - timedelta(days=1)
        return int(last_month_end.replace(day=1).timestamp())
    if "这个月" in query or "本月" in query:
        return int(today.replace(day=1).timestamp())

    m = _RECENT_RE.search(query)
    if m:
        return int((today - timedelta(days=int(m.group(1)))).timestamp())
    m = _AGO_RE.search(query)
    if m:
        return int((today - timedelta(days=int(m.group(1)))).timestamp())

    return None
