"""Fail-closed source-task allocation registry for Experiment 12.

The tracked JSON file is the scientific allocation.  This module validates its
exact schema and the experiment-specific invariants that JSON Schema alone
cannot express: cross-stage disjointness, outcome-blind exclusions, smoke-wave
membership, and ordered non-overlapping structural-failure reserves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import read_json, sha256_file
from experiments12.spec12 import Benchmark, Stage


SOURCE_REGISTRY_VERSION = 1
SOURCE_REGISTRY_TYPE = "experiment12_source_allocation_registry"
SOURCE_REGISTRY_PATH = Path(__file__).with_name("source_allocation12.json")
SOURCE_REGISTRY_SCHEMA_PATH = Path(__file__).with_name(
    "source_allocation12.schema.json"
)
CANONICAL_REALIZED_ALLOCATION_RECEIPTS: Mapping[tuple[str, str], Path] = {
    (
        Benchmark.EVOLVING_GSM8K.value,
        Stage.BASELINE_GATE.value,
    ): Path(__file__).with_name("evolving_baseline_screen12.json"),
    (
        Benchmark.EVOLVING_GSM8K.value,
        Stage.CALIBRATION.value,
    ): Path(__file__).with_name("evolving_calibration_screen12.json"),
    (
        Benchmark.EVOLVING_GSM8K.value,
        Stage.CONFIRMATORY.value,
    ): Path(__file__).with_name("evolving_confirmatory_screen12.json"),
    (
        Benchmark.EVOLVING_GSM8K.value,
        "deployment",
    ): Path(__file__).with_name("evolving_deployment_screen12.json"),
}

_STAGES = (
    Stage.SMOKE.value,
    Stage.BASELINE_GATE.value,
    Stage.CALIBRATION.value,
    Stage.CONFIRMATORY.value,
    "deployment",
)
_BENCHMARKS = tuple(item.value for item in (
    Benchmark.EVOLVING_GSM8K,
    Benchmark.BFCL_MULTI_TURN,
))
_TOP_FIELDS = {
    "$schema",
    "schema_version",
    "artifact_type",
    "selection_frozen_before_target_outcomes",
    "cross_stage_reuse_forbidden",
    "benchmarks",
}
_BENCHMARK_FIELDS = {
    "source_namespace",
    "allocation_rule",
    "allocations",
    "diagnostic_exclusions",
    "structural_failure_reserve",
}
_ALLOCATION_FIELDS = {"source_ids", "waves"}
_EXCLUSION_FIELDS = {"source_id", "reason_code"}
_REASON_CODES = {
    "prior_generation_diagnostic",
    "structural_screen_rejection",
}
_BFCL_ID = re.compile(
    r"^multi_turn_(?:base|miss_func|miss_param|long_context)_[0-9]+$"
)


class SourceRegistryError(ValueError):
    """The source allocation is malformed or a task selection escapes it."""


@dataclass(frozen=True, slots=True)
class SourceAllocationBinding:
    registry_sha256: str
    benchmark: str
    stage: str
    wave: str | None
    source_ids: tuple[str, ...]
    realized_allocation_sha256: str | None = None
    structural_rejection_source_ids: tuple[str, ...] = ()
    replacement_source_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "registry_sha256": self.registry_sha256,
            "benchmark": self.benchmark,
            "stage": self.stage,
            "wave": self.wave,
            "source_ids": list(self.source_ids),
            "realized_allocation_sha256": self.realized_allocation_sha256,
            "structural_rejection_source_ids": list(
                self.structural_rejection_source_ids
            ),
            "replacement_source_ids": list(self.replacement_source_ids),
        }


def _exact(value: Any, fields: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SourceRegistryError(f"{where} has missing or unexpected fields")
    return value


def _text(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(character in value for character in "\x00\r\n")
    ):
        raise SourceRegistryError(f"{where} must be bounded nonempty text")
    return value


def _source_ids(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SourceRegistryError(f"{where} must be a nonempty source-ID array")
    result = tuple(_text(item, f"{where}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise SourceRegistryError(f"{where} contains duplicate source IDs")
    return result


def validate_source_registry(value: Any) -> dict[str, Any]:
    root = _exact(value, _TOP_FIELDS, "source registry")
    if (
        root["$schema"] != SOURCE_REGISTRY_SCHEMA_PATH.name
        or root["schema_version"] != SOURCE_REGISTRY_VERSION
        or root["artifact_type"] != SOURCE_REGISTRY_TYPE
        or root["selection_frozen_before_target_outcomes"] is not True
        or root["cross_stage_reuse_forbidden"] is not True
    ):
        raise SourceRegistryError("source registry identity/freeze flags changed")
    benchmarks = root["benchmarks"]
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != set(_BENCHMARKS):
        raise SourceRegistryError("source registry must contain exactly the two core benchmarks")

    for benchmark in _BENCHMARKS:
        item = _exact(
            benchmarks[benchmark], _BENCHMARK_FIELDS, f"benchmarks.{benchmark}"
        )
        _text(item["source_namespace"], f"benchmarks.{benchmark}.source_namespace")
        _text(item["allocation_rule"], f"benchmarks.{benchmark}.allocation_rule")
        allocations = item["allocations"]
        if not isinstance(allocations, Mapping) or tuple(allocations) != _STAGES:
            raise SourceRegistryError(
                f"benchmarks.{benchmark}.allocations must use the frozen stage order"
            )
        allocated: set[str] = set()
        allocation_sizes: dict[str, int] = {}
        for stage in _STAGES:
            allocation = _exact(
                allocations[stage],
                _ALLOCATION_FIELDS,
                f"benchmarks.{benchmark}.allocations.{stage}",
            )
            ids = _source_ids(
                allocation["source_ids"],
                f"benchmarks.{benchmark}.allocations.{stage}.source_ids",
            )
            overlap = allocated.intersection(ids)
            if overlap:
                raise SourceRegistryError(
                    f"{benchmark} source is allocated to multiple stages: {sorted(overlap)[0]}"
                )
            allocated.update(ids)
            allocation_sizes[stage] = len(ids)
            waves = allocation["waves"]
            if not isinstance(waves, Mapping) or any(
                not isinstance(key, str) or not key for key in waves
            ):
                raise SourceRegistryError(f"{benchmark}/{stage} waves must be an object")
            if stage != Stage.SMOKE.value and waves:
                raise SourceRegistryError(f"only smoke may declare nested waves: {benchmark}/{stage}")
            for wave, raw_wave_ids in waves.items():
                wave_ids = _source_ids(raw_wave_ids, f"{benchmark}/{stage}/{wave}")
                if not set(wave_ids).issubset(ids):
                    raise SourceRegistryError(f"{benchmark}/{wave} escapes its smoke allocation")

        exclusions = item["diagnostic_exclusions"]
        if not isinstance(exclusions, list):
            raise SourceRegistryError(f"{benchmark} diagnostic_exclusions must be an array")
        excluded: list[str] = []
        for index, raw in enumerate(exclusions):
            row = _exact(raw, _EXCLUSION_FIELDS, f"{benchmark}.diagnostic_exclusions[{index}]")
            source_id = _text(row["source_id"], f"{benchmark}.exclusion.source_id")
            if row["reason_code"] not in _REASON_CODES:
                raise SourceRegistryError(f"{benchmark} exclusion has an unknown reason")
            excluded.append(source_id)
        if len(excluded) != len(set(excluded)):
            raise SourceRegistryError(f"{benchmark} exclusions contain duplicates")
        reserve = _source_ids(
            item["structural_failure_reserve"],
            f"benchmarks.{benchmark}.structural_failure_reserve",
        )
        if allocated.intersection(excluded) or allocated.intersection(reserve):
            raise SourceRegistryError(f"{benchmark} allocation overlaps exclusion/reserve")
        if set(excluded).intersection(reserve):
            raise SourceRegistryError(f"{benchmark} exclusion overlaps structural reserve")

        expected_sizes = {
            Benchmark.EVOLVING_GSM8K.value: (5, 20, 20, 56, 40),
            Benchmark.BFCL_MULTI_TURN.value: (8, 20, 20, 56, 40),
        }[benchmark]
        if tuple(allocation_sizes[stage] for stage in _STAGES) != expected_sizes:
            raise SourceRegistryError(f"{benchmark} frozen stage sample sizes changed")
        all_ids = allocated | set(excluded) | set(reserve)
        if benchmark == Benchmark.EVOLVING_GSM8K.value:
            if any(not source_id.isdecimal() for source_id in all_ids):
                raise SourceRegistryError("Evolving source IDs must be decimal GSM8K indices")
            smoke = allocations[Stage.SMOKE.value]
            if smoke["source_ids"] != ["12", "36", "40", "43", "50"]:
                raise SourceRegistryError("Evolving accepted smoke facts changed")
            if smoke["waves"] != {
                "single_model": ["12", "36", "40"],
                "all_models": ["12", "36", "40", "43", "50"],
            }:
                raise SourceRegistryError("Evolving smoke waves changed")
            if exclusions != [
                {"source_id": "14", "reason_code": "prior_generation_diagnostic"},
                {"source_id": "16", "reason_code": "prior_generation_diagnostic"},
                {"source_id": "49", "reason_code": "structural_screen_rejection"},
            ]:
                raise SourceRegistryError("Evolving diagnostic exclusions changed")
        else:
            if any(_BFCL_ID.fullmatch(source_id) is None for source_id in all_ids):
                raise SourceRegistryError("BFCL source IDs do not use the frozen namespace")
            expected_all_model = [
                "multi_turn_base_3",
                "multi_turn_base_4",
                "multi_turn_miss_func_0",
                "multi_turn_miss_param_0",
                "multi_turn_long_context_0",
            ]
            if allocations[Stage.SMOKE.value]["waves"].get("all_models") != expected_all_model:
                raise SourceRegistryError("BFCL all-model smoke selection changed")
    return dict(root)


def load_source_registry(path: str | Path = SOURCE_REGISTRY_PATH) -> dict[str, Any]:
    return validate_source_registry(read_json(path))


def normalize_source_id(benchmark: str, value: str) -> str:
    raw = _text(value, "source task ID").split("::", 1)[0]
    if benchmark == Benchmark.EVOLVING_GSM8K.value:
        prefix = "extracted-gsm8k-test-"
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
        if not raw.isdecimal():
            raise SourceRegistryError(f"invalid Evolving source-task identity: {value!r}")
        return str(int(raw))
    if benchmark == Benchmark.BFCL_MULTI_TURN.value and _BFCL_ID.fullmatch(raw):
        return raw
    raise SourceRegistryError(f"invalid source-task identity for {benchmark}: {value!r}")


def task_row_source_ids(rows: Sequence[Mapping[str, Any]], benchmark: str) -> tuple[str, ...]:
    if not rows:
        raise SourceRegistryError("task selection is empty")
    result: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("benchmark") != benchmark:
            raise SourceRegistryError(f"task row {index} belongs to another benchmark")
        raw = row.get("source_task_id", row.get("task_id"))
        if not isinstance(raw, str):
            raise SourceRegistryError(f"task row {index} lacks source_task_id")
        result.add(normalize_source_id(benchmark, raw))
    return tuple(sorted(result))


def _realized_allocation(
    *,
    path: str | Path,
    benchmark: str,
    stage_name: str,
    primary_ids: tuple[str, ...],
    reserve_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    """Validate an outcome-blind structural screen against registry order."""

    receipt_path = Path(path)
    raw = read_json(receipt_path)
    if not isinstance(raw, Mapping):
        raise SourceRegistryError("realized-allocation receipt must be an object")
    required = {
        "schema_version",
        "purpose",
        "candidate_source_ids_in_order",
        "target_valid_tasks",
        "maximum_attempts_per_candidate",
        "acceptance",
        "structural_rejections",
        "selected_source_ids",
    }
    if not required.issubset(raw):
        raise SourceRegistryError("realized-allocation receipt lacks required fields")
    if raw["schema_version"] != 1:
        raise SourceRegistryError("realized-allocation receipt version changed")
    purpose = raw["purpose"]
    if not isinstance(purpose, str) or stage_name.replace("_", "-") not in purpose:
        raise SourceRegistryError("realized-allocation receipt purpose/stage changed")
    if raw["target_valid_tasks"] != len(primary_ids):
        raise SourceRegistryError("realized allocation changed the frozen sample size")
    if raw["maximum_attempts_per_candidate"] != 1:
        raise SourceRegistryError("structural candidates must have one frozen attempt")
    acceptance = raw["acceptance"]
    if (
        not isinstance(acceptance, Mapping)
        or acceptance.get("target_model_outcomes_available_at_selection") is not False
    ):
        raise SourceRegistryError("realized allocation is not outcome-blind")

    def ids(value: Any, where: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise SourceRegistryError(f"{where} must be an array")
        normalized: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, (str, int)) or isinstance(item, bool):
                raise SourceRegistryError(f"{where}[{index}] is not a source ID")
            normalized.append(normalize_source_id(benchmark, str(item)))
        if len(normalized) != len(set(normalized)):
            raise SourceRegistryError(f"{where} contains duplicate source IDs")
        return tuple(normalized)

    candidates = ids(raw["candidate_source_ids_in_order"], "candidate source IDs")
    if len(candidates) < len(primary_ids) or candidates[: len(primary_ids)] != primary_ids:
        raise SourceRegistryError("realized candidates changed primary registry order")
    reserve_prefix = candidates[len(primary_ids) :]
    if reserve_prefix != reserve_ids[: len(reserve_prefix)]:
        raise SourceRegistryError("realized candidates skipped the ordered reserve")
    rejections_raw = raw["structural_rejections"]
    if not isinstance(rejections_raw, Mapping):
        raise SourceRegistryError("structural rejections must be an object")
    rejection_ids: list[str] = []
    for source_id, reason in rejections_raw.items():
        normalized = normalize_source_id(benchmark, str(source_id))
        if normalized not in candidates or not isinstance(reason, str) or not reason.strip():
            raise SourceRegistryError("structural rejection is invalid")
        rejection_ids.append(normalized)
    if len(rejection_ids) != len(set(rejection_ids)):
        raise SourceRegistryError("structural rejections contain duplicates")
    rejected = set(rejection_ids)
    expected = tuple(
        source_id
        for source_id in candidates
        if source_id not in rejected
    )[: len(primary_ids)]
    selected = ids(raw["selected_source_ids"], "selected source IDs")
    if selected != expected or len(selected) != len(primary_ids):
        raise SourceRegistryError(
            "realized selection is not the first valid registry-ordered candidates"
        )
    last_selected = candidates.index(selected[-1])
    if any(candidates.index(source_id) > last_selected for source_id in rejected):
        raise SourceRegistryError("receipt rejects unneeded candidates after selection filled")
    replacements = tuple(source_id for source_id in selected if source_id in reserve_ids)
    # ``selected == expected`` and the candidate-prefix check above already
    # prove that replacements use the reserve in order.  A reserve candidate
    # may itself fail the structural screen, so the selected replacements need
    # not be a literal prefix of ``reserve_ids``.
    return selected, tuple(rejection_ids), replacements, sha256_file(receipt_path)


def _reserve_before_stage(
    *,
    benchmark: str,
    stage_name: str,
    registry: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the globally ordered reserve remaining before ``stage_name``.

    Earlier tracked structural screens consume both accepted replacements and
    rejected reserve candidates.  This prevents a later split from silently
    reusing a reserve task that was already attempted or rejected.
    """

    remaining = tuple(registry["benchmarks"][benchmark]["structural_failure_reserve"])
    try:
        stage_index = _STAGES.index(stage_name)
    except ValueError as exc:
        raise SourceRegistryError(f"unsupported allocation stage: {stage_name}") from exc
    for prior_stage in _STAGES[:stage_index]:
        receipt = CANONICAL_REALIZED_ALLOCATION_RECEIPTS.get(
            (benchmark, prior_stage)
        )
        if receipt is None:
            continue
        if not receipt.is_file():
            raise SourceRegistryError(
                f"canonical realized-allocation receipt is missing: "
                f"{benchmark}/{prior_stage}"
            )
        primary = tuple(
            registry["benchmarks"][benchmark]["allocations"][prior_stage][
                "source_ids"
            ]
        )
        _selected, rejected, replacements, _digest = _realized_allocation(
            path=receipt,
            benchmark=benchmark,
            stage_name=prior_stage,
            primary_ids=primary,
            reserve_ids=remaining,
        )
        consumed_set = set(rejected).intersection(remaining) | set(replacements)
        consumed = tuple(source_id for source_id in remaining if source_id in consumed_set)
        if set(consumed) != consumed_set or consumed != remaining[: len(consumed)]:
            raise SourceRegistryError(
                "earlier realized allocations did not consume a contiguous reserve prefix"
            )
        remaining = remaining[len(consumed) :]
    return remaining


def bind_task_allocation(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: Stage | str,
    registry_path: str | Path = SOURCE_REGISTRY_PATH,
    smoke_wave: str | None = None,
    realized_allocation_path: str | Path | None = None,
) -> SourceAllocationBinding:
    stage_name = stage.value if isinstance(stage, Stage) else str(stage)
    if stage_name not in _STAGES:
        raise SourceRegistryError(f"unsupported allocation stage: {stage_name}")
    benchmark_values = {row.get("benchmark") for row in rows}
    if len(benchmark_values) != 1:
        raise SourceRegistryError("one task manifest must contain exactly one benchmark")
    benchmark = next(iter(benchmark_values))
    if benchmark not in _BENCHMARKS:
        raise SourceRegistryError("task manifest benchmark is outside the core registry")
    registry_file = Path(registry_path)
    registry = load_source_registry(registry_file)
    allocation = registry["benchmarks"][benchmark]["allocations"][stage_name]
    if smoke_wave is not None:
        if stage_name != Stage.SMOKE.value:
            raise SourceRegistryError("a smoke wave may only be used for stage=smoke")
        try:
            expected = tuple(allocation["waves"][smoke_wave])
        except KeyError as exc:
            raise SourceRegistryError(f"unknown smoke wave: {smoke_wave}") from exc
    else:
        expected = tuple(allocation["source_ids"])
    realization_sha256 = None
    rejection_ids: tuple[str, ...] = ()
    replacement_ids: tuple[str, ...] = ()
    if realized_allocation_path is not None:
        if smoke_wave is not None:
            raise SourceRegistryError("smoke waves cannot use a replacement receipt")
        canonical_receipt = CANONICAL_REALIZED_ALLOCATION_RECEIPTS.get(
            (str(benchmark), stage_name)
        )
        if (
            canonical_receipt is not None
            and sha256_file(realized_allocation_path) != sha256_file(canonical_receipt)
        ):
            raise SourceRegistryError(
                "realized allocation differs from the tracked canonical receipt"
            )
        expected, rejection_ids, replacement_ids, realization_sha256 = (
            _realized_allocation(
                path=realized_allocation_path,
                benchmark=str(benchmark),
                stage_name=stage_name,
                primary_ids=expected,
                reserve_ids=_reserve_before_stage(
                    benchmark=str(benchmark),
                    stage_name=stage_name,
                    registry=registry,
                ),
            )
        )
    observed = task_row_source_ids(rows, benchmark)
    if set(observed) != set(expected):
        raise SourceRegistryError(
            f"task sources differ from {benchmark}/{stage_name} allocation; "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    return SourceAllocationBinding(
        registry_sha256=sha256_file(registry_file),
        benchmark=benchmark,
        stage=stage_name,
        wave=smoke_wave,
        source_ids=expected,
        realized_allocation_sha256=realization_sha256,
        structural_rejection_source_ids=rejection_ids,
        replacement_source_ids=replacement_ids,
    )


def validate_source_allocation_binding(
    value: Any,
    *,
    registry_path: str | Path = SOURCE_REGISTRY_PATH,
    realized_allocation_path: str | Path | None = None,
) -> SourceAllocationBinding:
    """Reproduce a serialized binding, resolving only canonical receipts."""

    fields = {
        "registry_sha256",
        "benchmark",
        "stage",
        "wave",
        "source_ids",
        "realized_allocation_sha256",
        "structural_rejection_source_ids",
        "replacement_source_ids",
    }
    raw = _exact(value, fields, "source allocation binding")
    benchmark, stage_name, wave = raw["benchmark"], raw["stage"], raw["wave"]
    if not isinstance(benchmark, str) or not isinstance(stage_name, str):
        raise SourceRegistryError("source allocation binding identity is invalid")
    registry_file = Path(registry_path)
    registry = load_source_registry(registry_file)
    if raw["registry_sha256"] != sha256_file(registry_file):
        raise SourceRegistryError("source allocation binding registry hash changed")
    try:
        allocation = registry["benchmarks"][benchmark]["allocations"][stage_name]
    except KeyError as exc:
        raise SourceRegistryError("source allocation binding stage is absent") from exc
    if wave is not None:
        if stage_name != Stage.SMOKE.value or wave not in allocation["waves"]:
            raise SourceRegistryError("source allocation binding wave is invalid")
        primary = tuple(allocation["waves"][wave])
    else:
        primary = tuple(allocation["source_ids"])
    receipt_digest = raw["realized_allocation_sha256"]
    if receipt_digest is None:
        expected = primary
        rejections: tuple[str, ...] = ()
        replacements: tuple[str, ...] = ()
    else:
        if wave is not None:
            raise SourceRegistryError("smoke-wave bindings cannot contain replacements")
        receipt = (
            Path(realized_allocation_path)
            if realized_allocation_path is not None
            else CANONICAL_REALIZED_ALLOCATION_RECEIPTS.get((benchmark, stage_name))
        )
        if receipt is None or not receipt.is_file() or sha256_file(receipt) != receipt_digest:
            raise SourceRegistryError(
                "source allocation binding lacks its canonical realized receipt"
            )
        expected, rejections, replacements, reproduced_digest = _realized_allocation(
            path=receipt,
            benchmark=benchmark,
            stage_name=stage_name,
            primary_ids=primary,
            reserve_ids=_reserve_before_stage(
                benchmark=benchmark,
                stage_name=stage_name,
                registry=registry,
            ),
        )
        if reproduced_digest != receipt_digest:
            raise SourceRegistryError("realized-allocation receipt hash changed")
    if (
        raw["source_ids"] != list(expected)
        or raw["structural_rejection_source_ids"] != list(rejections)
        or raw["replacement_source_ids"] != list(replacements)
    ):
        raise SourceRegistryError("serialized source allocation binding does not reproduce")
    return SourceAllocationBinding(
        registry_sha256=raw["registry_sha256"],
        benchmark=benchmark,
        stage=stage_name,
        wave=wave,
        source_ids=expected,
        realized_allocation_sha256=receipt_digest,
        structural_rejection_source_ids=rejections,
        replacement_source_ids=replacements,
    )


__all__ = [
    "CANONICAL_REALIZED_ALLOCATION_RECEIPTS",
    "SOURCE_REGISTRY_PATH",
    "SOURCE_REGISTRY_SCHEMA_PATH",
    "SOURCE_REGISTRY_TYPE",
    "SOURCE_REGISTRY_VERSION",
    "SourceAllocationBinding",
    "SourceRegistryError",
    "bind_task_allocation",
    "load_source_registry",
    "normalize_source_id",
    "task_row_source_ids",
    "validate_source_registry",
    "validate_source_allocation_binding",
]
