"""Typed TypeSafe System One client.

Jev evaluates questions; it does not generate memory content. Requests share one
state across independently evaluated questions. A client-wide semaphore bounds
both batch fan-out and concurrent SDK writes.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dual_mem.providers.usage import UsageCallback, UsageEvent


class JevAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    choice: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    probabilities: dict[str, float] = Field(default_factory=dict)
    noul: float | None = Field(default=None, ge=0, le=1)
    score: float | None = None

    @model_validator(mode="after")
    def validate_answer(self) -> JevAnswer:
        if self.type == "choice":
            # Self-hosted System One servers (train-jev) return only choice +
            # probabilities; the chosen option's probability is its confidence.
            if (
                self.confidence is None
                and self.choice
                and self.choice in self.probabilities
            ):
                self.confidence = min(max(float(self.probabilities[self.choice]), 0.0), 1.0)
            if not self.choice or self.confidence is None:
                raise ValueError("choice answer requires choice and confidence")
            if self.choice not in self.probabilities:
                raise ValueError("choice is missing from probabilities")
        elif self.type == "noul":
            if self.noul is None:
                raise ValueError("noul answer requires probability")
        elif self.type == "score":
            if self.score is None:
                raise ValueError("score answer requires score")
        else:
            raise ValueError(f"unsupported answer type: {self.type}")
        if any(not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError("probabilities must be between zero and one")
        return self


class DecisionProvider(Protocol):
    async def evaluate(
        self, *, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, JevAnswer]: ...


class JevClient:
    """Pooled async HTTP client for the official /v1/systemone contract."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str = "jev-1.13.0",
        concurrency: int = 4,
        max_questions: int = 64,
        max_request_chars: int = 100_000,
        max_question_chars: int = 60_000,
        timeout: float = 30,
        http_client: httpx.AsyncClient | None = None,
        usage_callback: UsageCallback | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Jev requires jev_api_key or an injected decision provider"
            )
        if (
            min(
                concurrency,
                max_questions,
                max_request_chars,
                max_question_chars,
            )
            <= 0
            or timeout <= 0
        ):
            raise ValueError(
                "Jev limits, concurrency and timeout must be positive"
            )
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.endpoint = endpoint
        self.model = model
        if max_questions > 255:
            raise ValueError("Jev max_questions must be at most 255")
        self.max_questions = max_questions
        self.max_request_chars = max_request_chars
        self.max_question_chars = max_question_chars
        self._semaphore = asyncio.Semaphore(concurrency)
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.usage_callback = usage_callback
        self.request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_latency_ms = 0.0

    async def evaluate(
        self, *, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, JevAnswer]:
        if not questions:
            return {}
        state_chars = len(json.dumps(state, ensure_ascii=False))
        batches: list[dict[str, dict[str, Any]]] = []
        batch: dict[str, dict[str, Any]] = {}
        size = state_chars
        for name, question in questions.items():
            if question.get("type") not in ("choice", "noul", "score"):
                raise ValueError("unsupported Jev question type")
            if (
                question["type"] == "choice"
                and not 1 <= len(question.get("criteria", {})) <= 255
            ):
                raise ValueError("Jev choice must have 1-255 options")
            extra = len(json.dumps({name: question}, ensure_ascii=False))
            if state_chars + extra > self.max_question_chars:
                raise ValueError(
                    "Jev state + question exceeds context budget; split materials"
                )
            if batch and (
                len(batch) >= self.max_questions
                or size + extra > self.max_request_chars
            ):
                batches.append(batch)
                batch, size = {}, state_chars
            if size + extra > self.max_request_chars:
                raise ValueError("Jev request exceeds context budget")
            batch[name] = question
            size += extra
        if batch:
            batches.append(batch)
        results = await asyncio.gather(
            *(self._request(state=state, questions=part) for part in batches)
        )
        return {key: answer for part in results for key, answer in part.items()}

    async def _request(
        self, *, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, JevAnswer]:
        async with self._semaphore:
            started = time.perf_counter()
            self.request_count += 1
            response = None
            usage: dict = {}
            success = False
            try:
                response = await self._client.post(
                    self.endpoint, headers=self._headers,
                    json={"model": self.model, "state": state, "questions": questions},
                )
                response.raise_for_status()
                payload = response.json()
                raw = payload.get("answers")
                if not isinstance(raw, dict) or set(raw) != set(questions):
                    raise ValueError("Jev response must cover exactly the requested questions")
                answers = {name: JevAnswer.model_validate(value) for name, value in raw.items()}
                for name, answer in answers.items():
                    question = questions[name]
                    if answer.type != question["type"]:
                        raise ValueError("Jev answer type does not match question")
                    if answer.type == "choice" and answer.choice not in question["criteria"]:
                        raise ValueError("Jev selected an unknown option")
                usage = payload.get("usage") or {}
                self.input_tokens += int(usage.get("input_tokens", 0))
                self.output_tokens += int(usage.get("output_tokens", 0))
                success = True
                return answers
            finally:
                self.last_latency_ms = (time.perf_counter() - started) * 1000
                if self.usage_callback:
                    self.usage_callback(UsageEvent(
                        kind="jev", model=self.model, latency_ms=self.last_latency_ms,
                        prompt_tokens=int(usage.get("input_tokens", 0)),
                        completion_tokens=int(usage.get("output_tokens", 0)),
                        batch_size=len(questions), success=success, attempt=1,
                        status_code=response.status_code if response is not None else None,
                    ))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
