"""Deterministic, answer-blind passive quizzes from completed public turns."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from experiments12.monitors.frozen_quiz import QuizQuestion


_EVOLVING_TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]*|-?\d+", re.IGNORECASE)


def generate_evolving_passive_quiz(
    task_records: Sequence[Mapping[str, Any]],
    after_turn: int,
) -> tuple[QuizQuestion, ...]:
    """Build a quiz from completed Evolving user messages, never gold/future data."""

    if isinstance(after_turn, bool) or not isinstance(after_turn, int) or after_turn < 1:
        raise ValueError("after_turn must be a positive integer")
    if isinstance(task_records, (str, bytes)) or not isinstance(task_records, Sequence):
        raise ValueError("task_records must be a sequence")
    prefix = tuple(task_records[:after_turn])
    if len(prefix) != after_turn:
        raise ValueError("after_turn exceeds the completed Evolving prefix")
    public_user_text: list[str] = []
    for expected_turn, record in enumerate(prefix, 1):
        if (
            not isinstance(record, Mapping)
            or record.get("event") != "task_turn"
            or record.get("task_turn") != expected_turn
        ):
            raise ValueError("Evolving task_records must be contiguous completed turns")
        user = record.get("user_message")
        if not isinstance(user, Mapping) or user.get("role") != "user":
            raise ValueError("Evolving task record lacks its public user message")
        content = user.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Evolving public user message must be nonempty text")
        public_user_text.append(content)

    latest_tokens = tuple(
        match.group(0).lower()
        for match in _EVOLVING_TOKEN_RE.finditer(public_user_text[-1])
    )
    first_tokens = tuple(
        match.group(0).lower()
        for match in _EVOLVING_TOKEN_RE.finditer(public_user_text[0])
    )
    questions: list[QuizQuestion] = [
        QuizQuestion(
            f"evolving_completed_turns_{after_turn}",
            "How many benchmark user turns have been completed so far?",
            "integer",
            after_turn,
        ),
        QuizQuestion(
            f"evolving_latest_token_count_{after_turn}",
            "How many word/number tokens are in the latest completed benchmark user "
            "message, treating punctuation as separators?",
            "integer",
            len(latest_tokens),
        ),
    ]
    if latest_tokens:
        questions.append(
            QuizQuestion(
                f"evolving_latest_first_tokens_{after_turn}",
                "List the first up to four word/number tokens from the latest completed "
                "benchmark user message, in order.",
                "ordered",
                latest_tokens[:4],
            )
        )
    if after_turn > 1 and first_tokens:
        questions.append(
            QuizQuestion(
                f"evolving_initial_first_tokens_{after_turn}",
                "List the first up to four word/number tokens from the first benchmark "
                "user message, in order.",
                "ordered",
                first_tokens[:4],
            )
        )
    return tuple(questions)


def generate_bfcl_passive_quiz(
    task_records: Sequence[Mapping[str, Any]],
    after_turn: int,
) -> tuple[QuizQuestion, ...]:
    """Generate questions solely from already observed public BFCL results."""

    if isinstance(after_turn, bool) or not isinstance(after_turn, int) or after_turn < 1:
        raise ValueError("after_turn must be a positive integer")
    if isinstance(task_records, (str, bytes)) or not isinstance(task_records, Sequence):
        raise ValueError("task_records must be a sequence")
    prefix = tuple(task_records[:after_turn])
    if len(prefix) != after_turn:
        raise ValueError("after_turn exceeds the observed task-record prefix")
    names: list[str] = []
    statuses: list[str] = []
    for expected_turn, record in enumerate(prefix, 1):
        if (
            not isinstance(record, Mapping)
            or record.get("event") != "task_turn"
            or record.get("task_turn") != expected_turn
        ):
            raise ValueError("task_records must be contiguous completed BFCL turns")
        results = record.get("tool_results", ())
        if not isinstance(results, list):
            raise ValueError("BFCL tool_results must be a list")
        for result in results:
            if not isinstance(result, Mapping):
                raise ValueError("BFCL tool result must be an object")
            name, status = result.get("name"), result.get("status")
            if not isinstance(name, str) or not name or not isinstance(status, str) or not status:
                raise ValueError("BFCL tool result lacks public name/status")
            names.append(name)
            statuses.append(status)
    questions: list[QuizQuestion] = [
        QuizQuestion(
            f"bfcl_completed_turns_{after_turn}",
            "How many BFCL user turns have been completed so far?",
            "integer",
            after_turn,
        ),
        QuizQuestion(
            f"bfcl_tool_results_{after_turn}",
            "How many official tool results have been observed so far?",
            "integer",
            len(names),
        ),
    ]
    if names:
        questions.extend(
            (
                QuizQuestion(
                    f"bfcl_tool_names_{after_turn}",
                    "Which tool names have appeared in official results so far? "
                    "List them alphabetically, separated by commas.",
                    "exact",
                    ", ".join(sorted(set(names))),
                ),
                QuizQuestion(
                    f"bfcl_latest_status_{after_turn}",
                    "What was the status of the latest official tool result?",
                    "exact",
                    statuses[-1],
                ),
            )
        )
    return tuple(questions)


__all__ = ["generate_bfcl_passive_quiz", "generate_evolving_passive_quiz"]
