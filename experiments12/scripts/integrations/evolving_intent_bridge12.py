#!/usr/bin/env python3
"""Offline JSON bridge for the Behavioral Sentinels Evolving Intent build.

This checkout-local adapter executes the pinned upstream GSM8K extraction,
counterfactual, predecessor, and rule-based rendering code.  It never creates a
provider client.  Instead, every upstream ``generate_json`` / ``generate_text``
call is replayed from explicit state or emitted to the parent process as one
model-call request.

The file is intentionally self-contained and standard-library-only.  It is an
untracked reproduction patch, not part of the upstream project.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import types
from typing import Any


PROTOCOL = "behavioral-sentinels.evolving-intent-bridge.v1"
CONTRACT_SHA256 = "f55cf9f207606eb5cd7e278ef4ce23e8815babc5afc87c7877350c92ff9f1f0e"
UPSTREAM_COMMIT = "993d6be9597ac03854b46362ccd647eb1bfd267a"
STATE_VERSION = 1
SEED = 42
STAGES = (
    "intent_extraction",
    "counterfactual_generation",
    "predecessor_generation",
)
RUNTIME_DEPENDENCIES = {
    "antlr4-python3-runtime": "4.13.2",
    "latex2sympy2-extended": "1.11.0",
    "math-verify": "0.9.0",
    "mpmath": "1.3.0",
    "sympy": "1.14.0",
}
ROOT = Path(__file__).resolve().parent

EXTRACTION_PROMPTS = (
    "intent_construction/intent_extraction/dataset_impl/gsm8k/prompts/"
)
COUNTERFACTUAL_ROOT = (
    "intent_construction/retrospective_expansion/counterfactual"
)
PREDECESSOR_ROOT = (
    "intent_construction/retrospective_expansion/predecessor"
)


class BridgeInputError(ValueError):
    """The parent supplied an invalid or incompatible bridge request."""


class StageFailure(RuntimeError):
    """Pinned upstream code exhausted its own validation/retry policy."""


class NeedModelCall(BaseException):
    """Control-flow signal that deliberately bypasses upstream Exception catches."""

    def __init__(self, call: dict[str, Any], fingerprint: str):
        super().__init__(call["call_key"])
        self.call = call
        self.fingerprint = fingerprint


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_request_envelope(request: Any) -> Mapping[str, Any]:
    if not isinstance(request, Mapping):
        raise BridgeInputError("request must be a JSON object")
    if request.get("protocol") != PROTOCOL:
        raise BridgeInputError("protocol mismatch")
    if request.get("contract_sha256") != CONTRACT_SHA256:
        raise BridgeInputError("contract hash mismatch")
    if request.get("seed") != SEED:
        raise BridgeInputError("seed must be 42")
    supplied_commit = request.get("upstream_commit")
    if supplied_commit is not None and supplied_commit != UPSTREAM_COMMIT:
        raise BridgeInputError("upstream commit mismatch")
    return request


def _normalized_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise BridgeInputError("upstream messages must be a non-empty sequence")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise BridgeInputError("upstream message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise BridgeInputError("upstream message has invalid role/content")
        result.append({"role": role, "content": content})
    return result


def _safe_step(value: Any) -> str:
    raw = str(value or "generation")
    return "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in raw)[:48]


def _json_call_metadata(stage: str, step: str) -> tuple[str, list[str]]:
    mapping = {
        "extraction-decompose": (
            "generator",
            [EXTRACTION_PROMPTS + "segmentation.txt"],
        ),
        "extraction-conversational": (
            "generator",
            [EXTRACTION_PROMPTS + "conversational.txt"],
        ),
        "extraction-verification": (
            "judge",
            [EXTRACTION_PROMPTS + "verification.txt"],
        ),
        "llm-judge": (
            "judge",
            [EXTRACTION_PROMPTS + "llm_judge.txt"],
        ),
        "generate-counterfactual-argument": (
            "generator",
            [COUNTERFACTUAL_ROOT + "/prompts/generate_counterfactual_math.txt"],
        ),
        "predecessor-function-generation": (
            "generator",
            [PREDECESSOR_ROOT + "/prompts/generate_predecessor_gsm8k.txt"],
        ),
        "cross-turn-relevance-check": (
            "judge",
            [PREDECESSOR_ROOT + "/prompts/cross_turn_relevance_check.txt"],
        ),
        "regenerate-arguments-feedback": (
            "generator",
            [PREDECESSOR_ROOT + "/generate_predecessors.py"],
        ),
    }
    if step not in mapping:
        raise BridgeInputError(f"unmapped upstream JSON step: {stage}/{step}")
    return mapping[step]


def _text_call_metadata(stage: str, max_tokens: int | None) -> tuple[str, list[str]]:
    if stage == "intent_extraction":
        return (
            "judge",
            [
                "intent_construction/intent_extraction/dataset_impl/gsm8k/"
                "extractor.py"
            ],
        )
    if stage == "predecessor_generation" and max_tokens == 10:
        return (
            "judge",
            [PREDECESSOR_ROOT + "/prompts/similarity_check_gsm8k.txt"],
        )
    if stage == "predecessor_generation":
        return ("judge", [PREDECESSOR_ROOT + "/generate_predecessors.py"])
    raise BridgeInputError(f"unmapped upstream text call in {stage}")


class ReplayTransport:
    """Deterministically replay or emit calls made by one upstream stage."""

    def __init__(
        self,
        *,
        stage: str,
        source_hash: str,
        prior_hash: str,
        state: Any,
        model_result: Any,
    ) -> None:
        self.stage = stage
        self.source_hash = source_hash
        self.prior_hash = prior_hash
        self.records: list[dict[str, Any]] = []
        pending: Mapping[str, Any] | None = None

        if state is not None:
            if not isinstance(state, Mapping):
                raise BridgeInputError("bridge state must be an object or null")
            if (
                state.get("version") != STATE_VERSION
                or state.get("stage") != stage
                or state.get("source_sha256") != source_hash
                or state.get("prior_sha256") != prior_hash
            ):
                raise BridgeInputError("bridge state does not match this stage input")
            raw_records = state.get("records")
            if not isinstance(raw_records, list):
                raise BridgeInputError("bridge replay records must be an array")
            for record in raw_records:
                if not isinstance(record, Mapping):
                    raise BridgeInputError("bridge replay record must be an object")
                call_key = record.get("call_key")
                fingerprint = record.get("fingerprint")
                text = record.get("text")
                digest = record.get("output_sha256")
                if (
                    not isinstance(call_key, str)
                    or not isinstance(fingerprint, str)
                    or not isinstance(text, str)
                    or digest != _text_sha256(text)
                ):
                    raise BridgeInputError("bridge replay record failed integrity checks")
                self.records.append(dict(record))
            pending_value = state.get("pending")
            if pending_value is not None:
                if not isinstance(pending_value, Mapping):
                    raise BridgeInputError("pending bridge call must be an object")
                pending = pending_value

        if model_result is None:
            if pending is not None:
                raise BridgeInputError("pending bridge call requires model_result")
        else:
            if pending is None or not isinstance(model_result, Mapping):
                raise BridgeInputError("model_result has no matching pending call")
            call_key = model_result.get("call_key")
            text = model_result.get("text")
            if call_key != pending.get("call_key") or not isinstance(text, str):
                raise BridgeInputError("model_result does not match pending call")
            digest = _text_sha256(text)
            supplied_digest = model_result.get("output_sha256")
            if supplied_digest is not None and supplied_digest != digest:
                raise BridgeInputError("model_result output hash mismatch")
            self.records.append(
                {
                    "call_key": call_key,
                    "fingerprint": pending.get("fingerprint"),
                    "text": text,
                    "output_sha256": digest,
                }
            )

        self.cursor = 0

    def _state_for(self, need: NeedModelCall) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "stage": self.stage,
            "source_sha256": self.source_hash,
            "prior_sha256": self.prior_hash,
            "records": self.records,
            "pending": {
                "call_key": need.call["call_key"],
                "fingerprint": need.fingerprint,
            },
        }

    def _consume_or_emit(
        self,
        *,
        kind: str,
        messages: Any,
        step: str,
        role: str,
        prompt_files: list[str],
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        attempt: int,
    ) -> str:
        normalized = _normalized_messages(messages)
        fingerprint = _sha256(
            {
                "kind": kind,
                "stage": self.stage,
                "step": step,
                "messages": normalized,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "attempt": attempt,
                "prompt_files": prompt_files,
            }
        )
        if self.cursor < len(self.records):
            record = self.records[self.cursor]
            if record.get("fingerprint") != fingerprint:
                raise BridgeInputError(
                    "upstream call sequence diverged from saved replay state"
                )
            self.cursor += 1
            return record["text"]

        call_key = (
            f"{self.stage}:{self.cursor + 1:03d}:{_safe_step(step)}:a{attempt + 1}"
        )
        output_limit = max_tokens if max_tokens is not None else 8192
        call = {
            "call_key": call_key,
            "role": role,
            "messages": normalized,
            "temperature": temperature,
            "max_output_tokens": output_limit,
            "output_schema": None,
            "prompt_files": prompt_files,
        }
        raise NeedModelCall(call, fingerprint)

    def generate_json(
        self,
        messages: Any,
        model: str = "gpt-4o",
        step: str = "generation",
        max_retries: int = 3,
        temperature: float = 0.7,
        rate_limit_retries: int = 10,
        reasoning_effort: str | None = None,
    ) -> Any:
        del model, rate_limit_retries
        role, prompt_files = _json_call_metadata(self.stage, step)
        last_error: json.JSONDecodeError | None = None
        for attempt in range(max_retries):
            text = self._consume_or_emit(
                kind="json",
                messages=messages,
                step=step,
                role=role,
                prompt_files=prompt_files,
                temperature=temperature,
                max_tokens=None,
                reasoning_effort=reasoning_effort,
                attempt=attempt,
            )
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def generate_text(
        self,
        messages: Any,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        del model, max_retries
        role, prompt_files = _text_call_metadata(self.stage, max_tokens)
        return self._consume_or_emit(
            kind="text",
            messages=messages,
            step="text-generation",
            role=role,
            prompt_files=prompt_files,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            attempt=0,
        )

    def assert_fully_consumed(self) -> None:
        if self.cursor != len(self.records):
            raise BridgeInputError("saved replay state contains unused model responses")


class _InlineFuture:
    def __init__(self, function: Any, *args: Any, **kwargs: Any) -> None:
        self._result: Any = None
        self._error: BaseException | None = None
        try:
            self._result = function(*args, **kwargs)
        except BaseException as exc:  # NeedModelCall must survive the future boundary.
            self._error = exc

    def result(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._result


class _InlineExecutor:
    """Deterministic substitute for counterfactual argument-level threading."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *args: Any) -> None:
        del args

    def submit(self, function: Any, *args: Any, **kwargs: Any) -> _InlineFuture:
        return _InlineFuture(function, *args, **kwargs)


def _inline_as_completed(futures: Any) -> list[Any]:
    return list(futures)


def _install_import_guards() -> None:
    """Make accidental SDK use impossible and avoid optional CLI dependencies."""

    openai_stub = types.ModuleType("openai")

    class _NetworkDisabled:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("provider clients are disabled inside experiment12_bridge")

    openai_stub.OpenAI = _NetworkDisabled
    openai_stub.AzureOpenAI = _NetworkDisabled
    sys.modules["openai"] = openai_stub

    tqdm_stub = types.ModuleType("tqdm")

    def _tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return iterable

    tqdm_stub.tqdm = _tqdm
    sys.modules["tqdm"] = tqdm_stub


def _patch_llm_utils(replay: ReplayTransport) -> Any:
    _install_import_guards()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    llm_utils = importlib.import_module(
        "intent_construction.intent_extraction.core.llm_utils"
    )
    llm_utils.generate_json = replay.generate_json
    llm_utils.generate_text = replay.generate_text
    return llm_utils


def _stage_intent_extraction(
    source: Mapping[str, Any], replay: ReplayTransport
) -> dict[str, Any]:
    _patch_llm_utils(replay)
    module = importlib.import_module(
        "intent_construction.intent_extraction.dataset_impl.gsm8k.extractor"
    )
    module.generate_json = replay.generate_json
    module.generate_text = replay.generate_text
    extractor = module.GSM8kExtractor(
        model="gpt-5.1",
        num_arguments=4,
        max_verification_attempts=5,
        verif_model="gpt-5.1",
        enable_model_verification=True,
    )
    sample = {
        "id": source["source_id"],
        "task": "math",
        "split": "test",
        "question": source["question"],
        "answer": source["answer"],
    }
    artifact = extractor.extract(sample)
    if not isinstance(artifact, dict):
        raise StageFailure("upstream intent extraction failed validation")
    if str(artifact.get("task_id")) != str(source["task_id"]):
        raise StageFailure("upstream extraction changed the official task ID")
    return artifact


def _stage_counterfactual(
    prior: Mapping[str, Any], replay: ReplayTransport
) -> dict[str, Any]:
    _patch_llm_utils(replay)
    module = importlib.import_module(
        "intent_construction.retrospective_expansion.counterfactual."
        "generate_counterfactuals"
    )
    module.generate_json = replay.generate_json
    module.ThreadPoolExecutor = _InlineExecutor
    module.as_completed = _inline_as_completed
    generator = module.CounterfactualGenerator(
        model="gpt-5.1",
        prompts_dir=str(ROOT / COUNTERFACTUAL_ROOT / "prompts"),
        dataset_type="math",
        max_attempts=5,
        temperature=1.0,
    )
    artifact = generator.generate_counterfactuals(dict(prior), num_counterfactuals=4)
    if not isinstance(artifact, dict):
        raise StageFailure("upstream counterfactual generation failed validation")
    return artifact


def _stage_predecessor(
    prior: Mapping[str, Any], replay: ReplayTransport
) -> dict[str, Any]:
    llm_utils = _patch_llm_utils(replay)
    module = importlib.import_module(
        "intent_construction.retrospective_expansion.predecessor."
        "generate_predecessors"
    )
    module.generate_json = replay.generate_json
    module.generate_text = replay.generate_text
    generator = module.PredecessorGenerator(
        model="gpt-5.1",
        prompts_dir=str(ROOT / PREDECESSOR_ROOT / "prompts"),
        dataset_type="gsm8k",
        num_predecessors=3,
        chain_types=None,
        max_attempts=5,
        temperature=1.0,
        reasoning_effort=None,
        fallback_model="gpt-5.1",
        share_num=None,
        max_verify_attempts=2,
        judge_model="gpt-5.1",
        verify_independence=True,
        independence_runs=3,
        max_independence_retries=2,
    )
    # Upstream's CLI seeds the module RNG, but the generator creates a separate
    # Random() instance.  Seed the intended instance so subprocess replay is stable.
    generator._rng.seed(SEED)
    # Pinned HEAD's GSM8K similarity prompt names its placeholders goal_a and
    # goal_b, while _llm_similarity_check formats function_a and function_b.
    # That upstream mismatch raises KeyError before every similarity call and
    # makes predecessor generation impossible.  Repair only the placeholder
    # names; the exact upstream prompt text and decision code remain otherwise
    # unchanged.  The bridge hash and capabilities receipt disclose this patch.
    generator.similarity_prompt_template = (
        generator.similarity_prompt_template
        .replace("{goal_a}", "{function_a}")
        .replace("{goal_b}", "{function_b}")
    )
    try:
        math_verifier = importlib.import_module(
            "intent_construction.intent_extraction.core.math_verifier"
        )
    except ImportError:
        math_verifier = None
    if math_verifier is not None:
        math_verifier.generate_text = replay.generate_text
    llm_utils.generate_json = replay.generate_json
    llm_utils.generate_text = replay.generate_text
    artifact = generator.generate_predecessors(dict(prior), num_predecessors=3)
    if not isinstance(artifact, dict):
        raise StageFailure("upstream predecessor generation failed validation")
    return artifact


def _advance_stage(request: Mapping[str, Any]) -> dict[str, Any]:
    stage = request.get("stage")
    if stage not in STAGES:
        raise BridgeInputError("unknown stage")
    source = request.get("source_task")
    prior = request.get("prior_artifacts")
    if not isinstance(source, Mapping) or not isinstance(prior, Mapping):
        raise BridgeInputError("source_task and prior_artifacts must be objects")
    for key in ("source_id", "task_id", "question", "answer"):
        if key not in source:
            raise BridgeInputError(f"source_task is missing {key}")
    expected_prior = {
        "intent_extraction": set(),
        "counterfactual_generation": {"intent_extraction"},
        "predecessor_generation": {
            "intent_extraction",
            "counterfactual_generation",
        },
    }[stage]
    if set(prior) != expected_prior:
        raise BridgeInputError("prior stage artifacts do not match stage order")

    replay = ReplayTransport(
        stage=stage,
        source_hash=_sha256(dict(source)),
        prior_hash=_sha256(dict(prior)),
        state=request.get("state"),
        model_result=request.get("model_result"),
    )
    random.seed(SEED)
    try:
        if stage == "intent_extraction":
            artifact = _stage_intent_extraction(source, replay)
        elif stage == "counterfactual_generation":
            artifact = _stage_counterfactual(prior["intent_extraction"], replay)
        else:
            artifact = _stage_predecessor(
                prior["counterfactual_generation"], replay
            )
    except NeedModelCall as need:
        return {
            "status": "needs_model_call",
            "state": replay._state_for(need),
            "call": need.call,
        }
    replay.assert_fully_consumed()
    return {"status": "complete", "artifact": artifact}


def _render_one(
    data_path: Path,
    task_id: str,
    condition: Mapping[str, Any],
) -> dict[str, Any]:
    module = importlib.import_module("situated_simulation.user_simulation")
    simulator = module.EvolvingIntent(
        data_path=data_path,
        mode="eval",
        domain="math",
        ordering="interleaved",
        num_turns=condition["num_turns"],
        num_revisions=condition["num_revisions"],
        num_switches=condition["num_switches"],
        seed=SEED,
        task_ids=[task_id],
        naturalizer_model=None,
        recap_method=None,
        prefix_style=None,
    )
    if len(simulator) != 1:
        raise StageFailure("upstream simulator did not render exactly one sample")
    sample = simulator[0]
    turns = [
        turn["content"] for turn in sample.turns if turn.get("role") == "user"
    ]
    if len(turns) != condition["num_turns"]:
        raise StageFailure("upstream simulator could not satisfy requested turn count")
    return {"task_id": task_id, "turns": turns, "label": sample.label}


def _render_pair(request: Mapping[str, Any]) -> dict[str, Any]:
    source = request.get("source_task")
    artifacts = request.get("stage_artifacts")
    conditions = request.get("conditions")
    if (
        not isinstance(source, Mapping)
        or not isinstance(artifacts, Mapping)
        or not isinstance(conditions, Mapping)
    ):
        raise BridgeInputError("render inputs must be objects")
    if set(artifacts) != set(STAGES):
        raise BridgeInputError("render requires all three stage artifacts")
    expected = {
        "t1": {"num_turns": 1, "num_revisions": 0, "num_switches": 0},
        "t7": {"num_turns": 7, "num_revisions": 2, "num_switches": 2},
    }
    if dict(conditions) != expected:
        raise BridgeInputError("render conditions differ from the frozen protocol")
    final_artifact = artifacts.get("predecessor_generation")
    if not isinstance(final_artifact, Mapping):
        raise BridgeInputError("predecessor artifact must be an object")
    task_id = str(source.get("task_id", "")).strip()
    if not task_id or str(final_artifact.get("task_id")) != task_id:
        raise BridgeInputError("render task ID mismatch")

    _install_import_guards()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    with tempfile.TemporaryDirectory(prefix="evolving_intent_render_") as temp:
        data_path = Path(temp) / "predecessor.json"
        data_path.write_text(
            json.dumps([dict(final_artifact)], ensure_ascii=False),
            encoding="utf-8",
        )
        records = []
        for name in ("t1", "t7"):
            record = _render_one(data_path, task_id, expected[name])
            record["condition"] = name
            records.append(record)
    return {
        "status": "rendered",
        "simulator": {
            "kind": "rule_based",
            "module": "situated_simulation.user_simulation.EvolvingIntent",
            "seed": SEED,
        },
        "records": records,
    }


def _dispatch(raw_request: Any) -> dict[str, Any]:
    request = _require_request_envelope(raw_request)
    operation = request.get("operation")
    if operation == "capabilities":
        dependencies: dict[str, dict[str, Any]] = {}
        for name, expected_version in RUNTIME_DEPENDENCIES.items():
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = None
            dependencies[name] = {
                "available": version == expected_version,
                "version": version,
                "expected_version": expected_version,
            }
        math_module_available = importlib.util.find_spec("math_verify") is not None
        dependency_ready = all(
            details["available"] for details in dependencies.values()
        )
        return {
            "protocol": PROTOCOL,
            "contract_sha256": CONTRACT_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "transport_mode": "emit_requests_only",
            "stages": list(STAGES),
            "renderer": {
                "kind": "rule_based",
                "module": "situated_simulation.user_simulation.EvolvingIntent",
            },
            "runtime": {
                "python": platform.python_version(),
                "dependencies": dependencies,
                "math_verifier_mode": (
                    "math_verify"
                    if dependency_ready and math_module_available
                    else "string_normalization_fallback"
                ),
            },
            "compatibility_patches": [
                {
                    "id": "gsm8k_similarity_placeholder_names",
                    "upstream_prompt": "similarity_check_gsm8k.txt",
                    "repair": "goal_a/goal_b -> function_a/function_b",
                }
            ],
        }
    if operation == "advance_stage":
        return _advance_stage(request)
    if operation == "render_pair":
        return _render_pair(request)
    raise BridgeInputError("unknown operation")


def main() -> int:
    try:
        request = json.load(sys.stdin)
        # Upstream utilities print progress and warnings.  Keep stdout a pure
        # single-JSON protocol stream and route those diagnostics to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            response = _dispatch(request)
    except Exception as exc:
        response = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    json.dump(response, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
