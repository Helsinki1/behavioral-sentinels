"""One immutable passive-monitor contract for every Experiment 12 run.

The contract is deliberately data, not a collection of CLI defaults.  It is
embedded verbatim in each run manifest and hashed so a shadow artifact cannot
silently mix monitor methods, probe variants, quiz generators, or judge
settings from different runs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from experiments12.core.artifacts import sha256_json
from experiments12.monitors.frozen_probe import FROZEN_PROBE_VERSION
from experiments12.monitors.frozen_quiz import QUIZ_VERSION
from experiments12.monitors.judge import JUDGE_VERSION, JUDGE_MODEL_NAME
from experiments12.monitors.trace_rules import TRACE_RULE_VERSION
from experiments12.probes12 import CURRENT_COPY, RECOMPUTE


PASSIVE_MONITOR_SPEC_VERSION = 1
EVOLVING_QUIZ_GENERATOR_NAME = "evolving_public_turn_tokens"
EVOLVING_QUIZ_GENERATOR_VERSION = 1
BFCL_QUIZ_GENERATOR_NAME = "bfcl_public_tool_results"
BFCL_QUIZ_GENERATOR_VERSION = 1

_CANONICAL_SPEC: dict[str, Any] = {
    "version": PASSIVE_MONITOR_SPEC_VERSION,
    "required_methods": [
        "turn_clock",
        "context_use",
        "trace_rules",
        "frozen_probe",
        "frozen_quiz",
        "trace_judge",
    ],
    "checkpoint": {
        "schedule": "after_each_nonfinal_task_turn",
        "source": "clean_trajectory.checkpoint_turns",
        "every": 1,
        "actionable_offset_turns": 1,
    },
    "determinism": {
        "temperature": None,
        "fork_is_discarded": True,
        "inputs_use_only_completed_public_prefix": True,
        "reasoning_effort_by_target": {
            "gpt-oss-120b": "low",
            "deepseek-v4-flash-0731": "none",
            "qwen3p7-plus": "none",
            "gpt-5.6-luna": "medium",
            "gpt-5.6-terra": "medium",
        },
    },
    "baselines": {
        "turn_clock": {"version": 1, "formula": "completed_turns / task_horizon"},
        "context_use": {"version": 1, "formula": "input_tokens / context_window"},
    },
    "frozen_probe": {
        "version": FROZEN_PROBE_VERSION,
        "variants": [CURRENT_COPY, RECOMPUTE],
        "max_output_tokens": 192,
    },
    "frozen_quiz": {
        "grader_version": QUIZ_VERSION,
        "max_output_tokens": 320,
        "fire_at_wrong": 1,
        "generators": {
            "evolving_intent_gsm8k": {
                "name": EVOLVING_QUIZ_GENERATOR_NAME,
                "version": EVOLVING_QUIZ_GENERATOR_VERSION,
            },
            "bfcl_multi_turn": {
                "name": BFCL_QUIZ_GENERATOR_NAME,
                "version": BFCL_QUIZ_GENERATOR_VERSION,
            },
        },
    },
    "trace_rules": {
        "version": TRACE_RULE_VERSION,
        "fire_threshold": 0.65,
    },
    "trace_judge": {
        "version": JUDGE_VERSION,
        "enabled": True,
        "model": JUDGE_MODEL_NAME,
        "max_output_tokens": 320,
        "reasoning_effort": "low",
    },
}

PASSIVE_MONITOR_SPEC_SHA256 = sha256_json(_CANONICAL_SPEC)


def canonical_passive_monitor_spec() -> dict[str, Any]:
    """Return an independent copy suitable for an immutable manifest."""

    return deepcopy(_CANONICAL_SPEC)


def passive_monitor_manifest_binding() -> dict[str, Any]:
    return {
        "sha256": PASSIVE_MONITOR_SPEC_SHA256,
        "spec": canonical_passive_monitor_spec(),
    }


def validate_passive_monitor_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact canonical spec, rejecting missing *and* extra fields."""

    if not isinstance(value, Mapping):
        raise ValueError("passive monitor spec must be an object")
    materialized = deepcopy(dict(value))
    if materialized != _CANONICAL_SPEC:
        raise ValueError("passive monitor spec differs from the canonical frozen spec")
    if sha256_json(materialized) != PASSIVE_MONITOR_SPEC_SHA256:
        raise ValueError("passive monitor spec hash is internally inconsistent")
    return materialized


def passive_monitor_spec_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    binding = manifest.get("passive_monitor_spec")
    if not isinstance(binding, Mapping) or set(binding) != {"sha256", "spec"}:
        raise ValueError("run manifest lacks the exact passive monitor binding")
    if binding.get("sha256") != PASSIVE_MONITOR_SPEC_SHA256:
        raise ValueError("run manifest passive monitor hash changed")
    spec = binding.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError("run manifest passive monitor spec is invalid")
    materialized = validate_passive_monitor_spec(spec)
    if sha256_json(materialized) != binding["sha256"]:
        raise ValueError("run manifest passive monitor spec/hash mismatch")
    return materialized


def quiz_generator_spec(spec: Mapping[str, Any], domain: str) -> dict[str, Any]:
    validated = validate_passive_monitor_spec(spec)
    generators = validated["frozen_quiz"]["generators"]
    generator = generators.get(domain)
    if not isinstance(generator, dict):
        raise ValueError(f"no frozen passive quiz generator for domain {domain!r}")
    return deepcopy(generator)


def effective_passive_method_names(
    spec: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return the exact analysis names, expanding frozen-probe variants."""

    validated = validate_passive_monitor_spec(
        canonical_passive_monitor_spec() if spec is None else spec
    )
    names: list[str] = []
    for method in validated["required_methods"]:
        if method == "frozen_probe":
            names.extend(
                f"frozen_probe:{variant}"
                for variant in validated["frozen_probe"]["variants"]
            )
        else:
            names.append(method)
    return tuple(sorted(names))


def assert_passive_runtime_overrides(
    spec: Mapping[str, Any],
    *,
    run_judge: bool | None,
    judge_model: str | None,
) -> None:
    """Reject legacy/CLI settings that disagree with the manifest."""

    validated = validate_passive_monitor_spec(spec)
    judge = validated["trace_judge"]
    if run_judge is not None and run_judge is not judge["enabled"]:
        raise ValueError("runtime judge setting differs from the frozen passive spec")
    if judge_model is not None and judge_model != judge["model"]:
        raise ValueError("runtime judge model differs from the frozen passive spec")


__all__ = [
    "BFCL_QUIZ_GENERATOR_NAME",
    "BFCL_QUIZ_GENERATOR_VERSION",
    "EVOLVING_QUIZ_GENERATOR_NAME",
    "EVOLVING_QUIZ_GENERATOR_VERSION",
    "PASSIVE_MONITOR_SPEC_SHA256",
    "PASSIVE_MONITOR_SPEC_VERSION",
    "assert_passive_runtime_overrides",
    "canonical_passive_monitor_spec",
    "effective_passive_method_names",
    "passive_monitor_manifest_binding",
    "passive_monitor_spec_from_manifest",
    "quiz_generator_spec",
    "validate_passive_monitor_spec",
]
