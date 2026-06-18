"""意图分类 + query keyword 提取 + 相对时间解析。

意图分类忠实复现 hy_memory _retrieval/intent（NAV > CONCEPTUAL > FACTUAL）。
``parse_time_range`` 为 dual-mem 新增：legacy reader 不解析时间，这里补上常见
中文相对时间词 → ``created_after`` 时间戳的最小实现。
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
    return bool(_NAV_REGEX.search(query or ""))


def is_conceptual(query: str) -> bool:
    if not query:
        return False
    q_lower = query.lower()
    return any(trig in q_lower for trig in _CONCEPTUAL_TRIGGERS)


def classify_intent(query: str) -> str:
    """返回 NAVIGATIONAL / CONCEPTUAL / FACTUAL，优先级 NAV > CONCEPTUAL > FACTUAL。"""
    if is_navigational(query):
        return "NAVIGATIONAL"
    if is_conceptual(query):
        return "CONCEPTUAL"
    return "FACTUAL"


def extract_keywords(query: str) -> list[str]:
    """中英混合分词：英文/数字 token ≥3 且非停用词，汉字段 ≥2，去重保序，≤10。"""
    if not query:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for tok in _TOKEN_RE.findall(query):
        t = tok.lower() if tok.isascii() else tok
        if tok.isascii():
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


_RECENT_RE = re.compile(r"(?:最近|近|过去)\s*(\d+)\s*天")
_AGO_RE = re.compile(r"(\d+)\s*天前")


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_time_range(query: str, now: datetime | None = None) -> int | None:
    """识别常见中文相对时间词，返回 ``created_after``（unix 秒）；无时间词返回 None。"""
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
