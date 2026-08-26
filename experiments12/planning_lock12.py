"""Measured-baseline difficulty, cost, and sample-size locks for Experiment 12.

Calibration and confirmatory manifests must not be created from an optimistic
spreadsheet.  This module derives a high-percentile resource profile from
complete clean baseline trajectories, projects every declared target and
zero-carry monitor cell at frozen catalog prices, adds retry headroom, and
checks both the run-stage scope and the global provider ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments12.core.artifacts import (
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
)
from experiments12.core.budget import BudgetLedger
from experiments12.models12 import (
    CATALOG,
    JUDGE_MODEL_NAME,
    TARGET_MODEL_NAMES,
    estimate_call_upper_bound_usd,
)
from experiments12.passive_spec12 import effective_passive_method_names
from experiments12.pairing12 import JobCell
from experiments12.source_registry12 import (
    CANONICAL_REALIZED_ALLOCATION_RECEIPTS,
    SOURCE_REGISTRY_PATH,
    SourceAllocationBinding,
    bind_task_allocation,
    load_source_registry,
    normalize_source_id,
    validate_source_allocation_binding,
)
from experiments12.spec12 import ARMS, STAGE_PROVIDER_USD, Benchmark, Operator, Stage


BASELINE_PROFILE_VERSION = 2
BASELINE_PROFILE_TYPE = "experiment12_measured_baseline_resource_profile"
PROJECTION_LOCK_VERSION = 1
PROJECTION_LOCK_TYPE = "experiment12_cost_sample_size_projection_lock"
PROFILE_HEADROOM = Decimal("1.25")
RETRY_HEADROOM = Decimal("1.15")
ACTIVE_INPUT_MULTIPLIER = Decimal("2.25")
SHADOW_TARGET_INPUT_MULTIPLIER = Decimal("3.25")
JUDGE_INPUT_MULTIPLIER = Decimal("1.50")
DEFAULT_TARGET_FIRING_RATE = Decimal("0.20")
DEFAULT_DISCORDANCE = Decimal("0.30")
DEFAULT_MDE = Decimal("0.21")
DEFAULT_ALPHA = Decimal("0.05")
DEFAULT_POWER = Decimal("0.80")
DEFAULT_DEPLOYMENT_MDE = Decimal("0.25")
OBSERVER_EFFECT_DESIGN = "observer_effect"
DEPLOYMENT_DESIGN = "deployment"
_DESIGN_FAMILIES = {OBSERVER_EFFECT_DESIGN, DEPLOYMENT_DESIGN}

_PROFILE_FIELDS = {
    "schema_version",
    "artifact_type",
    "created_from_complete_clean_baseline_trajectories",
    "quantile_rule",
    "profile_headroom_multiplier",
    "source_manifest_sha256s",
    "profiles",
}
_SLICE_FIELDS = {
    "model",
    "benchmark",
    "condition",
    "source_allocation",
    "n_tasks",
    "n_success",
    "success_rate",
    "source_task_ids",
    "p95_calls",
    "p95_input_tokens",
    "p95_output_tokens",
    "p95_checkpoints",
    "planning_input_tokens",
    "planning_output_tokens",
}
_LOCK_FIELDS = {
    "schema_version",
    "artifact_type",
    "baseline_only",
    "stage",
    "allocation_stage",
    "design_family",
    "benchmark",
    "source_allocation",
    "baseline_profile_sha256",
    "baseline_manifest_sha256s",
    "design",
    "sample_size",
    "projection_method",
    "model_projections",
    "provider_projections",
}
_DESIGN_FIELDS = {
    "models",
    "arms",
    "operators",
    "replicates",
    "n_source_tasks",
    "n_cells",
    "passive_shadow_bundles",
}
_SAMPLE_FIELDS = {
    "statistical_unit",
    "planned_n_tasks",
    "alpha_two_sided",
    "power",
    "paired_discordance",
    "minimum_detectable_absolute_effect",
    "required_n_tasks",
    "target_firing_rate",
    "calibration_target_fire_count",
}
_MODEL_PROJECTION_FIELDS = {
    "model",
    "provider",
    "profile_condition",
    "planning_clean_input_tokens",
    "planning_clean_output_tokens",
    "planning_checkpoints",
    "trajectory_cells",
    "shadow_bundles",
    "projected_input_tokens",
    "projected_output_tokens",
    "projected_usd_before_retry_headroom",
}
_PROVIDER_PROJECTION_FIELDS = {
    "provider",
    "projected_usd",
    "stage_scope_cap_usd",
    "spent_usd_at_lock",
    "reserved_usd_at_lock",
    "remaining_operational_usd_at_lock",
    "fits_stage_scope",
    "fits_remaining_operational",
}


class PlanningLockError(ValueError):
    """A baseline profile, projection, sample size, or budget gate failed."""


@dataclass(frozen=True, slots=True)
class ScientificLaunchBinding:
    allocation: SourceAllocationBinding
    projection_lock_sha256: str
    projected_provider_usd: Mapping[str, str]
    required_n_tasks: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_allocation": self.allocation.as_dict(),
            "projection_lock_sha256": self.projection_lock_sha256,
            "projected_provider_usd": dict(self.projected_provider_usd),
            "required_n_tasks": self.required_n_tasks,
        }


def _exact(value: Any, fields: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PlanningLockError(f"{where} has missing or unexpected fields")
    return value


def _digest(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PlanningLockError(f"{where} must be lowercase SHA256")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlanningLockError(f"{where} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, where: str, *, maximum: Decimal | None = None) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise PlanningLockError(f"{where} must be decimal") from exc
    if not result.is_finite() or result < 0 or (maximum is not None and result > maximum):
        raise PlanningLockError(f"{where} is outside its valid range")
    return result


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_CEILING))


def _nearest_rank(values: Sequence[int], probability: Decimal) -> int:
    if not values:
        raise PlanningLockError("cannot profile an empty baseline slice")
    rank = max(1, math.ceil(float(probability * len(values))))
    return sorted(values)[rank - 1]


def _canonical_success_rate(n_success: int, n_tasks: int) -> str:
    """Return the exact canonical decimal rate for a frozen baseline slice."""

    if n_tasks < 1 or n_success < 0 or n_success > n_tasks:
        raise PlanningLockError("baseline success count is outside its valid range")
    return format(Decimal(n_success) / Decimal(n_tasks), "f")


def _profile_condition(benchmark: str) -> str:
    if benchmark == Benchmark.EVOLVING_GSM8K.value:
        return "t7"
    if benchmark == Benchmark.BFCL_MULTI_TURN.value:
        return "official_native_tools"
    raise PlanningLockError(f"unsupported projection benchmark: {benchmark}")


def _task_call_resources(trajectory: Mapping[str, Any], where: str) -> tuple[int, int, int]:
    records = trajectory.get("task_records")
    if not isinstance(records, list) or not records:
        raise PlanningLockError(f"{where} lacks task call records")
    input_tokens = output_tokens = calls = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or not isinstance(record.get("call"), Mapping):
            raise PlanningLockError(f"{where} task record {index} lacks accounting")
        call = record["call"]
        usage = call.get("usage")
        event_ids = call.get("call_event_ids")
        if not isinstance(usage, Mapping) or not isinstance(event_ids, list) or not event_ids:
            raise PlanningLockError(f"{where} task record {index} accounting is incomplete")
        input_tokens += _integer(
            usage.get("input_tokens"), f"{where} task record {index} input_tokens"
        )
        output_tokens += _integer(
            usage.get("output_tokens"), f"{where} task record {index} output_tokens"
        )
        calls += len(event_ids)
    return calls, input_tokens, output_tokens


def build_baseline_resource_profile(
    layouts: Sequence[Any],
    *,
    registry_path: str | Path = SOURCE_REGISTRY_PATH,
) -> dict[str, Any]:
    """Derive official success and p95 resource slices from clean baselines."""

    registry = load_source_registry(registry_path)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    manifest_hashes: list[str] = []
    if not layouts:
        raise PlanningLockError("at least one baseline run layout is required")
    for layout in layouts:
        manifest = read_json(layout.manifest)
        if manifest.get("stage") != Stage.BASELINE_GATE.value:
            raise PlanningLockError("resource profiles may use only baseline_gate runs")
        if sha256_file(layout.pairs) != manifest.get("pair_manifest_sha256"):
            raise PlanningLockError("baseline pair manifest hash mismatch")
        manifest_hashes.append(sha256_file(layout.manifest))
        cells = tuple(JobCell.from_dict(row) for row in read_jsonl(layout.pairs))
        if not cells or any(cell.arm != "clean" or cell.operator != "none" for cell in cells):
            raise PlanningLockError("baseline resource runs must contain only clean/none cells")
        expected_outputs = {f"{cell.cell_id}.json" for cell in cells}
        actual_outputs = {path.name for path in layout.trajectories.glob("*.json")}
        if actual_outputs != expected_outputs:
            raise PlanningLockError("baseline trajectories do not exactly cover declared cells")
        for cell in cells:
            trajectory = read_json(layout.trajectories / f"{cell.cell_id}.json")
            if (
                trajectory.get("complete") is not True
                or trajectory.get("arm") != "clean"
                or trajectory.get("model") != cell.pair_key.model
                or trajectory.get("domain") != cell.pair_key.domain
            ):
                raise PlanningLockError(f"baseline trajectory identity differs: {cell.cell_id}")
            condition = trajectory.get("condition")
            if not isinstance(condition, str) or not condition:
                raise PlanningLockError("baseline trajectory lacks condition")
            evaluation = trajectory.get("evaluation")
            if not isinstance(evaluation, Mapping) or not isinstance(
                evaluation.get("success"), bool
            ):
                raise PlanningLockError(
                    f"baseline trajectory lacks binary official success: {cell.cell_id}"
                )
            calls, input_tokens, output_tokens = _task_call_resources(
                trajectory, cell.cell_id
            )
            checkpoints = trajectory.get("checkpoint_turns")
            if not isinstance(checkpoints, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in checkpoints
            ):
                raise PlanningLockError("baseline checkpoint schedule is invalid")
            source_id = normalize_source_id(cell.pair_key.domain, cell.pair_key.task_id)
            grouped.setdefault(
                (cell.pair_key.model, cell.pair_key.domain, condition), []
            ).append(
                {
                    "source_id": source_id,
                    "success": evaluation["success"],
                    "calls": calls,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "checkpoints": len(checkpoints),
                }
            )

    profiles: list[dict[str, Any]] = []
    for (model, benchmark, condition), rows in sorted(grouped.items()):
        source_ids = tuple(sorted(row["source_id"] for row in rows))
        if len(source_ids) != len(set(source_ids)):
            raise PlanningLockError("baseline slice repeats a source task")
        primary = set(
            registry["benchmarks"][benchmark]["allocations"][Stage.BASELINE_GATE.value][
                "source_ids"
            ]
        )
        realized_path = (
            None
            if set(source_ids) == primary
            else CANONICAL_REALIZED_ALLOCATION_RECEIPTS.get(
                (benchmark, Stage.BASELINE_GATE.value)
            )
        )
        try:
            allocation = bind_task_allocation(
                [
                    {"benchmark": benchmark, "source_task_id": source_id}
                    for source_id in source_ids
                ],
                stage=Stage.BASELINE_GATE,
                registry_path=registry_path,
                realized_allocation_path=realized_path,
            )
        except Exception as exc:
            raise PlanningLockError(
                f"baseline profile source set differs from its frozen allocation: "
                f"{model}/{benchmark}/{condition}"
            ) from exc
        p95_calls = _nearest_rank([row["calls"] for row in rows], Decimal("0.95"))
        p95_input = _nearest_rank(
            [row["input_tokens"] for row in rows], Decimal("0.95")
        )
        p95_output = _nearest_rank(
            [row["output_tokens"] for row in rows], Decimal("0.95")
        )
        p95_checkpoints = _nearest_rank(
            [row["checkpoints"] for row in rows], Decimal("0.95")
        )
        n_success = sum(1 for row in rows if row["success"])
        profiles.append(
            {
                "model": model,
                "benchmark": benchmark,
                "condition": condition,
                "source_allocation": allocation.as_dict(),
                "n_tasks": len(rows),
                "n_success": n_success,
                "success_rate": _canonical_success_rate(n_success, len(rows)),
                "source_task_ids": list(source_ids),
                "p95_calls": p95_calls,
                "p95_input_tokens": p95_input,
                "p95_output_tokens": p95_output,
                "p95_checkpoints": p95_checkpoints,
                "planning_input_tokens": math.ceil(p95_input * PROFILE_HEADROOM),
                "planning_output_tokens": math.ceil(p95_output * PROFILE_HEADROOM),
            }
        )
    return validate_baseline_resource_profile(
        {
            "schema_version": BASELINE_PROFILE_VERSION,
            "artifact_type": BASELINE_PROFILE_TYPE,
            "created_from_complete_clean_baseline_trajectories": True,
            "quantile_rule": "nearest_rank_p95",
            "profile_headroom_multiplier": str(PROFILE_HEADROOM),
            "source_manifest_sha256s": sorted(set(manifest_hashes)),
            "profiles": profiles,
        },
        registry_path=registry_path,
    )


def validate_baseline_resource_profile(
    value: Any,
    *,
    registry_path: str | Path = SOURCE_REGISTRY_PATH,
) -> dict[str, Any]:
    root = _exact(value, _PROFILE_FIELDS, "baseline resource profile")
    if (
        root["schema_version"] != BASELINE_PROFILE_VERSION
        or root["artifact_type"] != BASELINE_PROFILE_TYPE
        or root["created_from_complete_clean_baseline_trajectories"] is not True
        or root["quantile_rule"] != "nearest_rank_p95"
        or _decimal(root["profile_headroom_multiplier"], "profile headroom")
        != PROFILE_HEADROOM
    ):
        raise PlanningLockError("baseline resource profile identity/method changed")
    manifest_hashes = root["source_manifest_sha256s"]
    if not isinstance(manifest_hashes, list) or not manifest_hashes:
        raise PlanningLockError("baseline profile has no source manifests")
    if manifest_hashes != sorted(set(manifest_hashes)):
        raise PlanningLockError("baseline manifest hashes must be sorted and unique")
    for digest in manifest_hashes:
        _digest(digest, "baseline source manifest")
    profiles = root["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise PlanningLockError("baseline profile slices are empty")
    registry = load_source_registry(registry_path)
    keys: list[tuple[str, str, str]] = []
    for index, raw in enumerate(profiles):
        row = _exact(raw, _SLICE_FIELDS, f"baseline profiles[{index}]")
        model, benchmark, condition = row["model"], row["benchmark"], row["condition"]
        if model not in TARGET_MODEL_NAMES or benchmark not in registry["benchmarks"]:
            raise PlanningLockError("baseline profile contains an unknown model/benchmark")
        if not isinstance(condition, str) or not condition:
            raise PlanningLockError("baseline profile condition is invalid")
        n_tasks = _integer(row["n_tasks"], "baseline n_tasks", minimum=1)
        n_success = _integer(row["n_success"], "baseline n_success", minimum=0)
        if n_success > n_tasks:
            raise PlanningLockError("baseline n_success exceeds n_tasks")
        if (
            not isinstance(row["success_rate"], str)
            or row["success_rate"]
            != _canonical_success_rate(n_success, n_tasks)
        ):
            raise PlanningLockError(
                "baseline success_rate does not reproduce n_success / n_tasks"
            )
        source_ids = row["source_task_ids"]
        if (
            not isinstance(source_ids, list)
            or source_ids != sorted(set(source_ids))
            or len(source_ids) != n_tasks
        ):
            raise PlanningLockError("baseline profile source IDs are invalid")
        try:
            allocation = validate_source_allocation_binding(
                row["source_allocation"], registry_path=registry_path
            )
        except Exception as exc:
            raise PlanningLockError(
                "baseline profile source allocation does not reproduce"
            ) from exc
        if (
            allocation.benchmark != benchmark
            or allocation.stage != Stage.BASELINE_GATE.value
            or set(source_ids) != set(allocation.source_ids)
        ):
            raise PlanningLockError(
                "baseline profile sources differ from its allocation binding"
            )
        for name in (
            "p95_calls",
            "p95_input_tokens",
            "p95_output_tokens",
            "p95_checkpoints",
            "planning_input_tokens",
            "planning_output_tokens",
        ):
            _integer(row[name], f"baseline {name}", minimum=0)
        if row["planning_input_tokens"] < math.ceil(
            row["p95_input_tokens"] * PROFILE_HEADROOM
        ) or row["planning_output_tokens"] < math.ceil(
            row["p95_output_tokens"] * PROFILE_HEADROOM
        ):
            raise PlanningLockError("baseline planning tokens omit frozen headroom")
        keys.append((model, benchmark, condition))
    if keys != sorted(set(keys)):
        raise PlanningLockError("baseline profile slices must be sorted and unique")
    return dict(root)


def freeze_baseline_resource_profile(
    path: str | Path,
    profile: Mapping[str, Any],
    *,
    registry_path: str | Path = SOURCE_REGISTRY_PATH,
) -> str:
    validated = validate_baseline_resource_profile(profile, registry_path=registry_path)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("baseline resource profile is write-once")
    return atomic_write_json(destination, validated)


def _sample_size(
    stage: Stage,
    n_tasks: int,
    *,
    design_family: str = OBSERVER_EFFECT_DESIGN,
) -> dict[str, Any]:
    if design_family not in _DESIGN_FAMILIES:
        raise PlanningLockError(f"unknown design family: {design_family}")
    mde = (
        DEFAULT_DEPLOYMENT_MDE
        if design_family == DEPLOYMENT_DESIGN
        else DEFAULT_MDE
    )
    z_sum = Decimal("1.959963984540054") + Decimal("0.8416212335729143")
    required = math.ceil(
        float((z_sum * z_sum * DEFAULT_DISCORDANCE) / (mde * mde))
    )
    target_fire_count = math.floor(float(DEFAULT_TARGET_FIRING_RATE * n_tasks))
    if stage is Stage.CALIBRATION:
        if n_tasks < 20 or target_fire_count < 4:
            raise PlanningLockError(
                "calibration requires at least 20 tasks and four fixed-rate firings"
            )
    elif stage is Stage.CONFIRMATORY and n_tasks < required:
        raise PlanningLockError(
            f"confirmatory n={n_tasks} is below the frozen paired-design requirement {required}"
        )
    return {
        "statistical_unit": "paired_source_task",
        "planned_n_tasks": n_tasks,
        "alpha_two_sided": str(DEFAULT_ALPHA),
        "power": str(DEFAULT_POWER),
        "paired_discordance": str(DEFAULT_DISCORDANCE),
        "minimum_detectable_absolute_effect": str(mde),
        "required_n_tasks": required,
        "target_firing_rate": str(DEFAULT_TARGET_FIRING_RATE),
        "calibration_target_fire_count": target_fire_count,
    }


def _slice_map(profile: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (row["model"], row["benchmark"], row["condition"]): row
        for row in profile["profiles"]
    }


def _cell_tokens(
    baseline: Mapping[str, Any], arm: str
) -> tuple[int, int]:
    clean_input = int(baseline["planning_input_tokens"])
    clean_output = int(baseline["planning_output_tokens"])
    checkpoints = int(baseline["p95_checkpoints"])
    if arm == "clean":
        return clean_input, clean_output
    if arm not in {item.name for item in ARMS if item.name != "clean"}:
        raise PlanningLockError(f"projection has an unknown active arm: {arm}")
    return (
        math.ceil(clean_input * ACTIVE_INPUT_MULTIPLIER),
        clean_output + checkpoints * 192,
    )


def _static_projection(
    *,
    profile: Mapping[str, Any],
    stage: Stage,
    design_family: str,
    benchmark: str,
    models: Sequence[str],
    arms: Sequence[str],
    operators: Sequence[str],
    replicates: int,
    n_tasks: int,
) -> tuple[list[dict[str, Any]], dict[str, Decimal]]:
    if stage not in {Stage.CALIBRATION, Stage.CONFIRMATORY}:
        raise PlanningLockError("projection locks are only for calibration/confirmatory")
    if design_family == OBSERVER_EFFECT_DESIGN:
        if tuple(arms) != ("clean", "active_recompute"):
            raise PlanningLockError(
                "observer-effect arm order is frozen to clean,active_recompute"
            )
        if tuple(operators) != ("none",) or replicates != 1:
            raise PlanningLockError(
                "observer-effect runs require operator=none, replicates=1"
            )
    elif design_family == DEPLOYMENT_DESIGN:
        active = {item.name for item in ARMS if item.name != "clean"}
        passive = set(effective_passive_method_names())
        if (
            not arms
            or len(arms) != len(set(arms))
            or set(arms) - active - passive
            or not set(arms).intersection(active)
            or not set(arms).intersection(passive)
        ):
            raise PlanningLockError(
                "deployment methods must be unique known methods with active and passive coverage"
            )
        valid_operators = {item.value for item in Operator}
        if (
            not operators
            or len(operators) != len(set(operators))
            or set(operators) - valid_operators
            or Operator.NONE.value not in operators
        ):
            raise PlanningLockError(
                "deployment operators must be unique known operators including none"
            )
        if replicates < 1:
            raise PlanningLockError("deployment replicates must be positive")
    else:
        raise PlanningLockError(f"unknown design family: {design_family}")
    if not models or len(models) != len(set(models)) or any(
        model not in TARGET_MODEL_NAMES for model in models
    ):
        raise PlanningLockError("projection models are empty, duplicated, or unknown")
    condition = _profile_condition(benchmark)
    slices = _slice_map(profile)
    provider_costs = {"openai": Decimal("0"), "fireworks": Decimal("0")}
    records: list[dict[str, Any]] = []
    for model in models:
        try:
            baseline = slices[(model, benchmark, condition)]
        except KeyError as exc:
            raise PlanningLockError(
                f"baseline profile lacks {model}/{benchmark}/{condition}"
            ) from exc
        checkpoints = int(baseline["p95_checkpoints"])
        before = Decimal("0")
        projected_input = projected_output = 0
        if design_family == OBSERVER_EFFECT_DESIGN:
            trajectory_bundles = n_tasks
            for arm in arms:
                input_tokens, output_tokens = _cell_tokens(baseline, arm)
                cost = estimate_call_upper_bound_usd(
                    model,
                    input_tokens,
                    output_tokens,
                    input_headroom=Decimal("0"),
                )
                before += cost * n_tasks
                projected_input += input_tokens * n_tasks
                projected_output += output_tokens * n_tasks
        else:
            # Deployment policies can fire active probes or passive monitors at
            # different checkpoints.  Cost every declared method x operator
            # cell as the more expensive active trajectory, then add both the
            # maximum target-model observer and judge below.  A deployment cell
            # selects one method, so costing both observer classes overstates it.
            trajectory_bundles = n_tasks * len(arms) * len(operators) * replicates
            input_tokens = math.ceil(
                int(baseline["planning_input_tokens"])
                * SHADOW_TARGET_INPUT_MULTIPLIER
            )
            output_tokens = (
                int(baseline["planning_output_tokens"]) + checkpoints * 192
            )
            before += estimate_call_upper_bound_usd(
                model,
                input_tokens,
                output_tokens,
                input_headroom=Decimal("0"),
            ) * trajectory_bundles
            projected_input += input_tokens * trajectory_bundles
            projected_output += output_tokens * trajectory_bundles

        shadow_input = math.ceil(
            int(baseline["planning_input_tokens"])
            * SHADOW_TARGET_INPUT_MULTIPLIER
        )
        shadow_output = checkpoints * (
            (192 * 2 + 320)
            if design_family == OBSERVER_EFFECT_DESIGN
            else 320
        )
        before += estimate_call_upper_bound_usd(
            model,
            shadow_input,
            shadow_output,
            input_headroom=Decimal("0"),
        ) * trajectory_bundles
        projected_input += shadow_input * trajectory_bundles
        projected_output += shadow_output * trajectory_bundles

        judge_input = (
            math.ceil(
                int(baseline["planning_input_tokens"])
                * JUDGE_INPUT_MULTIPLIER
            )
            + 4096
        )
        judge_output = checkpoints * 320
        judge_cost = estimate_call_upper_bound_usd(
            JUDGE_MODEL_NAME,
            judge_input,
            judge_output,
            input_headroom=Decimal("0"),
        ) * trajectory_bundles
        provider = CATALOG.models[model].provider
        provider_costs[provider] += before
        provider_costs["openai"] += judge_cost
        records.append(
            {
                "model": model,
                "provider": provider,
                "profile_condition": condition,
                "planning_clean_input_tokens": baseline["planning_input_tokens"],
                "planning_clean_output_tokens": baseline["planning_output_tokens"],
                "planning_checkpoints": checkpoints,
                "trajectory_cells": (
                    len(arms) * n_tasks
                    if design_family == OBSERVER_EFFECT_DESIGN
                    else trajectory_bundles
                ),
                "shadow_bundles": trajectory_bundles,
                "projected_input_tokens": projected_input,
                "projected_output_tokens": projected_output,
                "projected_usd_before_retry_headroom": _money(before + judge_cost),
            }
        )
    return records, {
        provider: cost * RETRY_HEADROOM for provider, cost in provider_costs.items()
    }


def build_projection_lock(
    *,
    baseline_profile_path: str | Path,
    registry_path: str | Path,
    ledger_path: str | Path,
    stage: Stage,
    benchmark: str,
    models: Sequence[str],
    arms: Sequence[str],
    operators: Sequence[str],
    replicates: int = 1,
    allocation_stage: Stage | str | None = None,
    design_family: str = OBSERVER_EFFECT_DESIGN,
    realized_allocation_path: str | Path | None = None,
) -> dict[str, Any]:
    profile_path = Path(baseline_profile_path)
    registry_file = Path(registry_path)
    profile = validate_baseline_resource_profile(
        read_json(profile_path), registry_path=registry_file
    )
    registry = load_source_registry(registry_file)
    if benchmark not in registry["benchmarks"]:
        raise PlanningLockError("projection benchmark is absent from source registry")
    allocation_name = (
        stage.value
        if allocation_stage is None
        else allocation_stage.value
        if isinstance(allocation_stage, Stage)
        else str(allocation_stage)
    )
    if design_family == DEPLOYMENT_DESIGN:
        if stage is not Stage.CONFIRMATORY or allocation_name != "deployment":
            raise PlanningLockError(
                "deployment locks require provider stage=confirmatory and allocation_stage=deployment"
            )
    elif design_family == OBSERVER_EFFECT_DESIGN:
        if allocation_name != stage.value:
            raise PlanningLockError(
                "observer-effect allocation stage must equal its provider stage"
            )
    else:
        raise PlanningLockError(f"unknown design family: {design_family}")
    try:
        allocation = registry["benchmarks"][benchmark]["allocations"][allocation_name]
    except KeyError as exc:
        raise PlanningLockError(
            f"projection allocation stage is absent: {allocation_name}"
        ) from exc
    source_ids = tuple(allocation["source_ids"])
    if realized_allocation_path is not None:
        receipt = read_json(realized_allocation_path)
        selected = receipt.get("selected_source_ids") if isinstance(receipt, Mapping) else None
        if not isinstance(selected, list):
            raise PlanningLockError(
                "realized-allocation receipt lacks selected_source_ids"
            )
        try:
            realized_binding = bind_task_allocation(
                [
                    {"benchmark": benchmark, "source_task_id": str(source_id)}
                    for source_id in selected
                ],
                stage=allocation_name,
                registry_path=registry_file,
                realized_allocation_path=realized_allocation_path,
            )
        except Exception as exc:
            raise PlanningLockError(
                "projection realized source allocation does not reproduce"
            ) from exc
        source_ids = realized_binding.source_ids
    n_tasks = len(source_ids)
    sample = _sample_size(stage, n_tasks, design_family=design_family)
    model_records, costs = _static_projection(
        profile=profile,
        stage=stage,
        design_family=design_family,
        benchmark=benchmark,
        models=tuple(models),
        arms=tuple(arms),
        operators=tuple(operators),
        replicates=replicates,
        n_tasks=n_tasks,
    )
    ledger = BudgetLedger(ledger_path)
    snapshots = ledger.snapshot()
    provider_records: list[dict[str, Any]] = []
    for provider in ("fireworks", "openai"):
        projected = costs[provider]
        scope_cap = Decimal(str(STAGE_PROVIDER_USD[stage][provider]))
        snapshot = snapshots[provider]
        fits_scope = projected <= scope_cap
        fits_operational = projected <= snapshot.remaining_operational_usd
        provider_records.append(
            {
                "provider": provider,
                "projected_usd": _money(projected),
                "stage_scope_cap_usd": _money(scope_cap),
                "spent_usd_at_lock": _money(snapshot.spent_usd),
                "reserved_usd_at_lock": _money(snapshot.reserved_usd),
                "remaining_operational_usd_at_lock": _money(
                    snapshot.remaining_operational_usd
                ),
                "fits_stage_scope": fits_scope,
                "fits_remaining_operational": fits_operational,
            }
        )
        if not fits_scope or not fits_operational:
            raise PlanningLockError(
                f"{provider} projected ${_money(projected)} exceeds "
                f"stage/remaining operational budget"
            )
    lock = {
        "schema_version": PROJECTION_LOCK_VERSION,
        "artifact_type": PROJECTION_LOCK_TYPE,
        "baseline_only": True,
        "stage": stage.value,
        "allocation_stage": allocation_name,
        "design_family": design_family,
        "benchmark": benchmark,
        "source_allocation": {
            "registry_sha256": sha256_file(registry_file),
            "source_ids": list(source_ids),
        },
        "baseline_profile_sha256": sha256_file(profile_path),
        "baseline_manifest_sha256s": list(profile["source_manifest_sha256s"]),
        "design": {
            "models": list(models),
            "arms": list(arms),
            "operators": list(operators),
            "replicates": replicates,
            "n_source_tasks": n_tasks,
            "n_cells": n_tasks * len(models) * len(arms) * len(operators) * replicates,
            "passive_shadow_bundles": (
                n_tasks * len(models) * replicates
                if design_family == OBSERVER_EFFECT_DESIGN
                else n_tasks
                * len(models)
                * len(arms)
                * len(operators)
                * replicates
            ),
        },
        "sample_size": sample,
        "projection_method": {
            "baseline_quantile": "nearest_rank_p95",
            "profile_headroom_multiplier": str(PROFILE_HEADROOM),
            "retry_headroom_multiplier": str(RETRY_HEADROOM),
            "active_input_multiplier": str(ACTIVE_INPUT_MULTIPLIER),
            "shadow_target_input_multiplier": str(SHADOW_TARGET_INPUT_MULTIPLIER),
            "judge_input_multiplier": str(JUDGE_INPUT_MULTIPLIER),
            "cached_input_discount_assumed": False,
            "all_frozen_output_limits_used_for_observers": True,
            "deployment_cell_policy": (
                "not_applicable"
                if design_family == OBSERVER_EFFECT_DESIGN
                else "every_cell_as_active_plus_max_target_and_judge_observer"
            ),
        },
        "model_projections": model_records,
        "provider_projections": provider_records,
    }
    return validate_projection_lock_static(
        lock,
        baseline_profile_path=profile_path,
        registry_path=registry_file,
        realized_allocation_path=realized_allocation_path,
    )


def validate_projection_lock_static(
    value: Any,
    *,
    baseline_profile_path: str | Path,
    registry_path: str | Path,
    realized_allocation_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _exact(value, _LOCK_FIELDS, "projection lock")
    if (
        root["schema_version"] != PROJECTION_LOCK_VERSION
        or root["artifact_type"] != PROJECTION_LOCK_TYPE
        or root["baseline_only"] is not True
    ):
        raise PlanningLockError("projection lock identity changed")
    try:
        stage = Stage(root["stage"])
    except (TypeError, ValueError) as exc:
        raise PlanningLockError("projection lock stage is invalid") from exc
    if stage not in {Stage.CALIBRATION, Stage.CONFIRMATORY}:
        raise PlanningLockError("projection lock has a non-scientific stage")
    allocation_stage = root["allocation_stage"]
    design_family = root["design_family"]
    if not isinstance(allocation_stage, str) or not allocation_stage:
        raise PlanningLockError("projection allocation stage is invalid")
    if design_family not in _DESIGN_FAMILIES:
        raise PlanningLockError("projection design family is invalid")
    if design_family == DEPLOYMENT_DESIGN:
        if stage is not Stage.CONFIRMATORY or allocation_stage != "deployment":
            raise PlanningLockError("deployment projection stage binding changed")
    elif allocation_stage != stage.value:
        raise PlanningLockError("observer-effect projection stage binding changed")
    benchmark = root["benchmark"]
    registry_file = Path(registry_path)
    profile_file = Path(baseline_profile_path)
    registry = load_source_registry(registry_file)
    profile = validate_baseline_resource_profile(
        read_json(profile_file), registry_path=registry_file
    )
    if root["baseline_profile_sha256"] != sha256_file(profile_file):
        raise PlanningLockError("projection lock baseline profile hash mismatch")
    if root["baseline_manifest_sha256s"] != profile["source_manifest_sha256s"]:
        raise PlanningLockError("projection lock baseline manifest set changed")
    allocation = _exact(
        root["source_allocation"], {"registry_sha256", "source_ids"}, "source allocation"
    )
    if allocation["registry_sha256"] != sha256_file(registry_file):
        raise PlanningLockError("projection lock source registry hash mismatch")
    try:
        primary_sources = registry["benchmarks"][benchmark]["allocations"][
            allocation_stage
        ]["source_ids"]
    except KeyError as exc:
        raise PlanningLockError("projection allocation stage is absent") from exc
    locked_sources = allocation["source_ids"]
    if not isinstance(locked_sources, list) or not locked_sources:
        raise PlanningLockError("projection lock source allocation is invalid")
    if locked_sources == primary_sources:
        expected_sources = primary_sources
    else:
        receipt = (
            Path(realized_allocation_path)
            if realized_allocation_path is not None
            else CANONICAL_REALIZED_ALLOCATION_RECEIPTS.get(
                (benchmark, allocation_stage)
            )
        )
        if receipt is None:
            raise PlanningLockError(
                "projection lock needs its realized-allocation receipt"
            )
        try:
            realized_binding = bind_task_allocation(
                [
                    {"benchmark": benchmark, "source_task_id": str(source_id)}
                    for source_id in locked_sources
                ],
                stage=allocation_stage,
                registry_path=registry_file,
                realized_allocation_path=receipt,
            )
        except Exception as exc:
            raise PlanningLockError(
                "projection lock realized source allocation does not reproduce"
            ) from exc
        expected_sources = list(realized_binding.source_ids)
    if locked_sources != expected_sources:
        raise PlanningLockError("projection lock source allocation changed")
    design = _exact(root["design"], _DESIGN_FIELDS, "projection design")
    models, arms, operators = design["models"], design["arms"], design["operators"]
    if any(not isinstance(value, list) for value in (models, arms, operators)):
        raise PlanningLockError("projection design arrays are invalid")
    replicates = _integer(design["replicates"], "projection replicates", minimum=1)
    n_tasks = len(expected_sources)
    if (
        design["n_source_tasks"] != n_tasks
        or design["n_cells"]
        != n_tasks * len(models) * len(arms) * len(operators) * replicates
        or design["passive_shadow_bundles"]
        != (
            n_tasks * len(models) * replicates
            if design_family == OBSERVER_EFFECT_DESIGN
            else n_tasks
            * len(models)
            * len(arms)
            * len(operators)
            * replicates
        )
    ):
        raise PlanningLockError("projection design counts changed")
    sample = _exact(root["sample_size"], _SAMPLE_FIELDS, "sample-size lock")
    if sample != _sample_size(stage, n_tasks, design_family=design_family):
        raise PlanningLockError("sample-size lock does not reproduce")
    expected_method = {
        "baseline_quantile": "nearest_rank_p95",
        "profile_headroom_multiplier": str(PROFILE_HEADROOM),
        "retry_headroom_multiplier": str(RETRY_HEADROOM),
        "active_input_multiplier": str(ACTIVE_INPUT_MULTIPLIER),
        "shadow_target_input_multiplier": str(SHADOW_TARGET_INPUT_MULTIPLIER),
        "judge_input_multiplier": str(JUDGE_INPUT_MULTIPLIER),
        "cached_input_discount_assumed": False,
        "all_frozen_output_limits_used_for_observers": True,
        "deployment_cell_policy": (
            "not_applicable"
            if design_family == OBSERVER_EFFECT_DESIGN
            else "every_cell_as_active_plus_max_target_and_judge_observer"
        ),
    }
    if root["projection_method"] != expected_method:
        raise PlanningLockError("projection methodology changed")
    expected_models, costs = _static_projection(
        profile=profile,
        stage=stage,
        design_family=design_family,
        benchmark=benchmark,
        models=tuple(models),
        arms=tuple(arms),
        operators=tuple(operators),
        replicates=replicates,
        n_tasks=n_tasks,
    )
    if root["model_projections"] != expected_models:
        raise PlanningLockError("model projections do not reproduce")
    providers = root["provider_projections"]
    if not isinstance(providers, list) or len(providers) != 2:
        raise PlanningLockError("provider projections must cover both providers")
    for index, row in enumerate(providers):
        parsed = _exact(row, _PROVIDER_PROJECTION_FIELDS, f"provider projection {index}")
        provider = parsed["provider"]
        if provider not in costs or parsed["projected_usd"] != _money(costs[provider]):
            raise PlanningLockError("provider projected cost does not reproduce")
        if parsed["stage_scope_cap_usd"] != _money(
            Decimal(str(STAGE_PROVIDER_USD[stage][provider]))
        ):
            raise PlanningLockError("provider stage scope changed")
        for name in (
            "spent_usd_at_lock",
            "reserved_usd_at_lock",
            "remaining_operational_usd_at_lock",
        ):
            _decimal(parsed[name], f"provider {name}")
        if parsed["fits_stage_scope"] is not True or parsed[
            "fits_remaining_operational"
        ] is not True:
            raise PlanningLockError("projection lock records a failed budget gate")
    if [row["provider"] for row in providers] != ["fireworks", "openai"]:
        raise PlanningLockError("provider projection order changed")
    return dict(root)


def freeze_projection_lock(
    path: str | Path,
    lock: Mapping[str, Any],
    *,
    baseline_profile_path: str | Path,
    registry_path: str | Path,
    realized_allocation_path: str | Path | None = None,
) -> str:
    validated = validate_projection_lock_static(
        lock,
        baseline_profile_path=baseline_profile_path,
        registry_path=registry_path,
        realized_allocation_path=realized_allocation_path,
    )
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("cost/sample-size projection lock is write-once")
    return atomic_write_json(destination, validated)


def validate_projection_lock_current_budget(
    value: Any,
    *,
    baseline_profile_path: str | Path,
    registry_path: str | Path,
    ledger_path: str | Path,
    realized_allocation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reproduce a lock and recheck it against the live shared ledger."""

    validated = validate_projection_lock_static(
        value,
        baseline_profile_path=baseline_profile_path,
        registry_path=registry_path,
        realized_allocation_path=realized_allocation_path,
    )
    stage = Stage(validated["stage"])
    snapshots = BudgetLedger(ledger_path).snapshot()
    for row in validated["provider_projections"]:
        provider = row["provider"]
        projected = _decimal(row["projected_usd"], "projected cost")
        if projected > Decimal(str(STAGE_PROVIDER_USD[stage][provider])):
            raise PlanningLockError(f"{provider} projection exceeds stage scope")
        if projected > snapshots[provider].remaining_operational_usd:
            raise PlanningLockError(
                f"{provider} projection exceeds current remaining operational budget"
            )
    return validated


def assert_scientific_launch(
    *,
    task_rows: Sequence[Mapping[str, Any]],
    stage: Stage,
    models: Sequence[str],
    arms: Sequence[str],
    operators: Sequence[str],
    replicates: int,
    ledger_path: str | Path,
    registry_path: str | Path | None,
    projection_lock_path: str | Path | None,
    baseline_profile_path: str | Path | None,
    smoke_wave: str | None = None,
    allocation_stage: Stage | str | None = None,
    design_family: str = OBSERVER_EFFECT_DESIGN,
    realized_allocation_path: str | Path | None = None,
) -> ScientificLaunchBinding | SourceAllocationBinding | None:
    """Gate initialization before any manifest, pair file, or provider call exists.

    ``stage`` selects the provider budget.  ``allocation_stage`` independently
    selects a registry split; production deployment therefore uses
    ``stage=CONFIRMATORY`` with ``allocation_stage="deployment"``.
    """

    allocation_name: Stage | str = stage if allocation_stage is None else allocation_stage

    if stage is Stage.SMOKE:
        if design_family != OBSERVER_EFFECT_DESIGN:
            raise PlanningLockError("deployment design locks are confirmatory-only")
        if registry_path is None:
            return None
        return bind_task_allocation(
            task_rows,
            stage=allocation_name,
            registry_path=registry_path,
            smoke_wave=smoke_wave,
            realized_allocation_path=realized_allocation_path,
        )
    if stage is Stage.OFFLINE:
        if (
            allocation_stage is not None
            or design_family != OBSERVER_EFFECT_DESIGN
            or realized_allocation_path is not None
        ):
            raise PlanningLockError("offline launch cannot bind a scientific design")
        return None
    if registry_path is None:
        raise PlanningLockError(f"stage={stage.value} requires a source allocation registry")
    allocation = bind_task_allocation(
        task_rows,
        stage=allocation_name,
        registry_path=registry_path,
        realized_allocation_path=realized_allocation_path,
    )
    if stage is Stage.BASELINE_GATE:
        if allocation.stage != Stage.BASELINE_GATE.value:
            raise PlanningLockError("baseline provider stage requires baseline allocation")
        if design_family != OBSERVER_EFFECT_DESIGN:
            raise PlanningLockError("baseline launch cannot use deployment design")
        return allocation
    if projection_lock_path is None or baseline_profile_path is None:
        raise PlanningLockError(
            f"stage={stage.value} requires baseline profile and cost/sample-size lock"
        )
    lock_path = Path(projection_lock_path)
    lock = validate_projection_lock_current_budget(
        read_json(lock_path),
        baseline_profile_path=baseline_profile_path,
        registry_path=registry_path,
        ledger_path=ledger_path,
        realized_allocation_path=realized_allocation_path,
    )
    design = lock["design"]
    if (
        lock["stage"] != stage.value
        or lock["allocation_stage"] != allocation.stage
        or lock["design_family"] != design_family
        or lock["benchmark"] != allocation.benchmark
        or design["models"] != list(models)
        or design["arms"] != list(arms)
        or design["operators"] != list(operators)
        or design["replicates"] != replicates
        or lock["source_allocation"]["source_ids"] != list(allocation.source_ids)
    ):
        raise PlanningLockError("runtime scientific design differs from projection lock")
    projections = {row["provider"]: row for row in lock["provider_projections"]}
    return ScientificLaunchBinding(
        allocation=allocation,
        projection_lock_sha256=sha256_file(lock_path),
        projected_provider_usd={
            provider: projections[provider]["projected_usd"]
            for provider in ("fireworks", "openai")
        },
        required_n_tasks=int(lock["sample_size"]["required_n_tasks"]),
    )


__all__ = [
    "BASELINE_PROFILE_TYPE",
    "BASELINE_PROFILE_VERSION",
    "PROJECTION_LOCK_TYPE",
    "PROJECTION_LOCK_VERSION",
    "DEPLOYMENT_DESIGN",
    "OBSERVER_EFFECT_DESIGN",
    "PlanningLockError",
    "ScientificLaunchBinding",
    "assert_scientific_launch",
    "build_baseline_resource_profile",
    "build_projection_lock",
    "freeze_baseline_resource_profile",
    "freeze_projection_lock",
    "validate_baseline_resource_profile",
    "validate_projection_lock_static",
    "validate_projection_lock_current_budget",
]
