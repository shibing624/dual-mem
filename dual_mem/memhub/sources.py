# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Read each CLI's own transcript store into complete user/assistant
turns and ingest them incrementally into the MemHub store.

Readers only use the structure each host writes (record types, roles, text
blocks, harness markup tags); they never guess meaning from wording:

- agentica    ``~/.agentica/projects/**/<session>.jsonl``  (type user/assistant)
- claude-code ``~/.claude/projects/**/<session>.jsonl``    (message.content text blocks)
- codex       ``~/.codex/sessions/**/rollout-*.jsonl``     (event_msg UserMessage/AgentMessage
  items — the real input, unlike response_item which also carries AGENTS.md)
- cursor      ``~/.cursor/projects/*/agent-transcripts/**/*.jsonl``
- hermes      ``~/.hermes/state.db`` table ``messages`` (cursor = message id)

A turn is the user's message plus the assistant's *last* text before the next
user message — the answer, not the tool-call narration in between.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from dual_mem.client import strip_memory_injection
from dual_mem.memhub.config import CursorStore

SOURCES = ("agentica", "claude-code", "codex", "cursor", "hermes")

_TEXT_BLOCK_TYPES = {"text", "input_text", "output_text", "Text"}
# Harness markup Claude Code writes into user records (reminders, slash-command echo).
_CLAUDE_MARKUP = (
    "system-reminder",
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
    "local-command-stderr",
    "local-command-caveat",
)
_CURSOR_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)


class HubWriter(Protocol):
    async def write(
        self,
        *,
        content: str = "",
        messages: list[dict] | None = None,
        session_id: str = "",
        memory_at: int | None = None,
        request_id: str,
        raw_only: bool = False,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    session_id: str
    at: int | None
    # Reader positions before and after this record; a cursor stores ``end``.
    start: int
    end: int


@dataclass
class Turn:
    session_id: str
    user: str
    assistant: str
    at: int | None
    start: int
    end: int

    def messages(self) -> list[dict[str, str]]:
        out = [{"role": "user", "content": self.user}]
        if self.assistant:
            out.append({"role": "assistant", "content": self.assistant})
        return out


def _iso_seconds(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") in _TEXT_BLOCK_TYPES
        and isinstance(block.get("text"), str)
    )


def _strip_tags(text: str, tags: tuple[str, ...]) -> str:
    names = "|".join(re.escape(t) for t in tags)
    return re.sub(rf"<({names})\b[^>]*>.*?</\1\s*>", "", text, flags=re.DOTALL)


# A record parser gets (record, state) and returns (role, text, session_id, at) or None.
# ``state`` carries per-file context such as the Codex session id from session_meta.
RecordParser = Callable[[dict, dict], "tuple[str, str, str, int | None] | None"]


def _parse_agentica(record: dict, state: dict) -> tuple[str, str, str, int | None] | None:
    if record.get("type") not in ("user", "assistant"):
        return None
    return (
        record["type"],
        _blocks_text(record.get("content")),
        record.get("session_id") or state["session_id"],
        _iso_seconds(record.get("timestamp")),
    )


def _parse_claude(record: dict, state: dict) -> tuple[str, str, str, int | None] | None:
    if record.get("type") not in ("user", "assistant") or record.get("isSidechain") or record.get("isMeta"):
        return None
    text = _blocks_text((record.get("message") or {}).get("content"))
    return (
        record["type"],
        _strip_tags(text, _CLAUDE_MARKUP),
        record.get("sessionId") or state["session_id"],
        _iso_seconds(record.get("timestamp")),
    )


def _parse_codex(record: dict, state: dict) -> tuple[str, str, str, int | None] | None:
    payload = record.get("payload") or {}
    if record.get("type") == "session_meta":
        state["session_id"] = payload.get("id") or state["session_id"]
        return None
    if record.get("type") != "event_msg":
        return None
    at = _iso_seconds(record.get("timestamp"))
    kind = payload.get("type")
    if kind == "item_completed":
        item = payload.get("item") or {}
        role = {"UserMessage": "user", "AgentMessage": "assistant"}.get(
            str(item.get("type"))
        )
        if role is None:
            return None
        return role, _blocks_text(item.get("content")), state["session_id"], at
    # Rollouts written before item events existed.
    role = {"user_message": "user", "agent_message": "assistant"}.get(str(kind))
    if role is None:
        return None
    return role, str(payload.get("message") or ""), state["session_id"], at


def _parse_cursor(record: dict, state: dict) -> tuple[str, str, str, int | None] | None:
    role = record.get("role")
    if role not in ("user", "assistant"):
        return None
    text = _blocks_text((record.get("message") or {}).get("content"))
    if role == "user":
        # Cursor wraps what the user typed in <user_query>; user records without
        # it carry harness context (tool / MCP catalogs).
        query = _CURSOR_QUERY_RE.search(text)
        if query is None:
            return None
        text = query.group(1)
    return role, text, state["session_id"], None


def _record(line: bytes) -> dict | None:
    """A complete JSONL line as an object; partial or malformed lines are None."""
    if not line.endswith(b"\n"):
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def _read_jsonl(
    path: Path, start: int, parse: RecordParser
) -> tuple[list[Message], int]:
    """Messages after byte ``start``; stops before a trailing line still being written."""
    if start > path.stat().st_size:
        start = 0  # the transcript was rewritten or truncated
    state: dict[str, Any] = {"session_id": path.stem}
    messages: list[Message] = []
    position = start
    with path.open("rb") as fp:
        if start:
            # Codex names the session in its first record; resumed reads need it too.
            head = _record(fp.readline())
            if head is not None:
                parse(head, state)
            fp.seek(start)
        for line in fp:
            if not line.endswith(b"\n"):
                break
            line_start, position = position, position + len(line)
            record = _record(line)
            parsed = parse(record, state) if record is not None else None
            if parsed is None:
                continue
            role, text, session_id, at = parsed
            text = strip_memory_injection(text).strip()
            if text:
                messages.append(Message(role, text, session_id, at, line_start, position))
    return messages, position


def _read_hermes(path: Path, start: int) -> tuple[list[Message], int]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, session_id, role, content, timestamp FROM messages "
            "WHERE id > ? AND role IN ('user', 'assistant') AND active = 1 "
            "AND _compressed_summary = 0 ORDER BY id",
            (start,),
        ).fetchall()
    finally:
        conn.close()
    messages = [
        Message(role, text, session_id, int(ts), row_id - 1, row_id)
        for row_id, session_id, role, content, ts in rows
        if (text := strip_memory_injection(content or "").strip())
    ]
    return messages, rows[-1][0] if rows else start


@dataclass(frozen=True)
class SourceSpec:
    name: str
    # Relative to $HOME.
    root: str
    discover: Callable[[Path], list[Path]]
    read: Callable[[Path, int], tuple[list[Message], int]]


def _jsonl_files(pattern: str, *, inside: str | None = None) -> Callable[[Path], list[Path]]:
    def discover(root: Path) -> list[Path]:
        files = [
            p
            for p in root.rglob(pattern)
            if "subagents" not in p.parts and (inside is None or inside in p.parts)
        ]
        return sorted(files, key=lambda p: p.stat().st_mtime_ns)

    return discover


SOURCE_SPECS: dict[str, SourceSpec] = {
    "agentica": SourceSpec(
        "agentica",
        ".agentica/projects",
        _jsonl_files("*.jsonl"),
        lambda p, s: _read_jsonl(p, s, _parse_agentica),
    ),
    "claude-code": SourceSpec(
        "claude-code",
        ".claude/projects",
        _jsonl_files("*.jsonl"),
        lambda p, s: _read_jsonl(p, s, _parse_claude),
    ),
    "codex": SourceSpec(
        "codex",
        ".codex/sessions",
        _jsonl_files("rollout-*.jsonl"),
        lambda p, s: _read_jsonl(p, s, _parse_codex),
    ),
    "cursor": SourceSpec(
        "cursor",
        ".cursor/projects",
        _jsonl_files("*.jsonl", inside="agent-transcripts"),
        lambda p, s: _read_jsonl(p, s, _parse_cursor),
    ),
    "hermes": SourceSpec(
        "hermes",
        ".hermes/state.db",
        lambda root: [root] if root.is_file() else [],
        _read_hermes,
    ),
}


def default_source_roots(home: Path | None = None) -> dict[str, Path]:
    base = home or Path.home()
    return {name: base / spec.root for name, spec in SOURCE_SPECS.items()}


def build_turns(messages: list[Message]) -> tuple[list[Turn], Turn | None]:
    """Group messages into turns; the last turn is returned apart as still open.

    Consecutive user messages form one request; an assistant message replaces the
    turn's answer, so the final text wins. Assistant text with no user message
    before it in this slice belongs to a turn that was already ingested.
    """
    turns: list[Turn] = []
    current: Turn | None = None
    for msg in messages:
        if msg.role == "user":
            if current and (current.assistant or current.session_id != msg.session_id):
                turns.append(current)
                current = None
            if current is None:
                current = Turn(msg.session_id, msg.text, "", msg.at, msg.start, msg.end)
            else:
                current.user = f"{current.user}\n{msg.text}"
                current.end = msg.end
        elif current is not None and current.session_id == msg.session_id:
            current.assistant = msg.text
            current.end = msg.end
    return [t for t in turns if t.assistant], current


def _batches(turns: list[Turn], size: int) -> Iterator[list[Turn]]:
    batch: list[Turn] = []
    for turn in turns:
        if batch and (len(batch) >= size or batch[-1].session_id != turn.session_id):
            yield batch
            batch = []
        batch.append(turn)
    if batch:
        yield batch


async def scan_file(
    client: HubWriter,
    *,
    source: str,
    path: Path,
    cursors: CursorStore,
    complete: bool = False,
    idle_seconds: int = 300,
    batch_turns: int = 5,
    max_turns: int = 0,
    raw_only: bool = False,
) -> int:
    """Ingest the turns appended to ``path`` since its cursor; returns turns written.

    The open last turn is held back (cursor stays before it) unless the caller
    knows it ended (``complete``: a Stop / run.completed hook) or the transcript
    has been idle for ``idle_seconds``. Each batch of up to ``batch_turns`` turns of
    one session is one write with a content-derived request id. The cursor moves
    only past batches the Core holds durably; an unreachable Core raises and the
    next scan resends from there (same ids, so nothing is stored twice).
    """
    spec = SOURCE_SPECS[source]
    key = str(path.resolve())
    with cursors.lock(source, key):
        cursor = cursors.get(source, key)
        messages, reached = spec.read(path, cursor.position if cursor else 0)
        turns, open_turn = build_turns(messages)
        ended = complete or time.time() - path.stat().st_mtime >= idle_seconds
        if open_turn and open_turn.assistant and ended:
            turns.append(open_turn)
            open_turn = None
        stop = open_turn.start if open_turn else reached
        if max_turns and len(turns) > max_turns:
            turns = turns[:max_turns]
            stop = turns[-1].end
        for batch in _batches(turns, batch_turns):
            messages_in = [m for turn in batch for m in turn.messages()]
            session_id = f"{source}:{batch[0].session_id}"
            request_id = hashlib.sha256(
                json.dumps(
                    [
                        source,
                        batch[0].session_id,
                        batch[0].start,
                        batch[-1].end,
                        messages_in,
                    ],
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()[:32]
            # Returns once the Core holds the evidence (it retries model work
            # itself); an unreachable Core raises here and the cursor stays.
            await client.write(
                messages=messages_in,
                session_id=session_id,
                memory_at=batch[-1].at,
                request_id=f"memhub:{request_id}",
                raw_only=raw_only,
            )
            cursors.advance(source, key, batch[-1].end, len(batch))
        cursors.advance(source, key, stop, 0)
    return len(turns)


async def scan_sources(
    client: HubWriter,
    *,
    hub_dir: str | Path | None = None,
    sources: list[str] | None = None,
    roots: dict[str, Path] | None = None,
    max_turns: int = 0,
    **scan_kwargs: Any,
) -> dict[str, dict[str, int]]:
    """Scan every transcript of ``sources``; ``max_turns`` caps this run in total."""
    roots = roots or default_source_roots()
    summary: dict[str, dict[str, int]] = {}
    remaining = max_turns
    with CursorStore(hub_dir) as cursors:
        for source in sources or SOURCES:
            files = SOURCE_SPECS[source].discover(roots[source]) if roots[source].exists() else []
            written = 0
            for path in files:
                if max_turns and remaining <= 0:
                    break
                n = await scan_file(
                    client,
                    source=source,
                    path=path,
                    cursors=cursors,
                    max_turns=remaining if max_turns else 0,
                    **scan_kwargs,
                )
                written += n
                remaining -= n
            summary[source] = {"files": len(files), "turns": written}
    return summary


def locate_transcript(
    source: str, payload: dict, roots: dict[str, Path] | None = None
) -> Path | None:
    """The transcript a host hook payload refers to.

    Claude Code / Codex / Cursor hooks carry ``transcript_path``. Agentica's
    ``run.completed`` notice carries only ``session_id``, which names its log file.
    """
    if payload.get("transcript_path"):
        return Path(payload["transcript_path"]).expanduser()
    root = (roots or default_source_roots())[source]
    if source == "hermes":
        return root
    session_id = payload.get("session_id")
    if not session_id or not root.is_dir():
        return None
    return next(root.rglob(f"{session_id}.jsonl"), None)
