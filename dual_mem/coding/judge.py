"""Jev scene choice using full available source, never a keyword rule."""

import json
from typing import Any

from dual_mem.coding.types import CodingDecisionError
from dual_mem.providers.jev import DecisionProvider, JevAnswer


async def evaluate_decisions(
    jev: DecisionProvider,
    *,
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
) -> dict[str, JevAnswer]:
    """Bound full evidence and validate injected providers as well as HTTP JEV."""
    state_chars = len(json.dumps(state, ensure_ascii=False))
    budget = min(
        getattr(jev, "max_question_chars", 60_000),
        getattr(jev, "max_request_chars", 100_000),
    )
    for name, question in questions.items():
        if not 1 <= len(question.get("criteria", {})) <= 255:
            raise CodingDecisionError("coding JEV choice budget exceeded")
        if state_chars + len(json.dumps({name: question}, ensure_ascii=False)) > budget:
            raise CodingDecisionError(
                "coding JEV context budget exceeded; split source or expand budget"
            )
    answers = await jev.evaluate(state=state, questions=questions)
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise CodingDecisionError("coding JEV response lacks exact question coverage")
    for name, answer in answers.items():
        try:
            answer = JevAnswer.model_validate(answer)
        except (TypeError, ValueError) as exc:
            raise CodingDecisionError("invalid coding JEV answer") from exc
        if answer.type != "choice" or answer.choice not in questions[name]["criteria"]:
            raise CodingDecisionError("coding JEV selected an unsupported answer")
        answers[name] = answer
    return answers


async def classify_messages_is_coding(
    messages: list[dict], jev: DecisionProvider
) -> bool:
    if not messages:
        return False
    answers = await evaluate_decisions(
        jev,
        state={"messages": messages},
        questions={
            "scene": {
                "type": "choice",
                "instructions": "Choose dominant scene of the entire conversation. Incidental tool use does not imply engineering.",
                "criteria": {
                    "CODING": "Engineering work and reusable implementation learning",
                    "CHAT": "User conversation or non-engineering task",
                },
            }
        },
    )
    return answers["scene"].choice == "CODING"
