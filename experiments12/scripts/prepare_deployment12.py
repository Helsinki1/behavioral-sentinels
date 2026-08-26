"""Freeze an answer-blind two-pass deployment from completed observation runs.

This coordinator is deliberately provider-free.  It verifies the calibration
and pass-one observation provenance, reduces active probes and passive shadows
to score/hash-only traces, checks global source-task disjointness, and writes
the complete deployment pair/threshold/schedule/manifest chain exactly once.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from experiments12.analysis12 import (
    _trace_from_dict,
    calibrate_thresholds,
    extract_run,
    load_threshold_artifact,
    make_threshold_artifact,
)
from experiments12.cli12 import (
    DEFAULT_ARTIFACTS,
    REPOSITORY_ROOT,
    _evolving_provenance_receipts,
)
from experiments12.core.artifacts import (
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.budget import BudgetLedger
from experiments12.deployment12 import (
    DEPLOYMENT_SCHEDULE_RECEIPT,
    PASS_ONE_RECEIPT,
    THRESHOLD_LOCK_RECEIPT,
    TWO_PASS_DEPLOYMENT_MODE,
    DeploymentArtifactError,
    DeploymentEstimand,
    PassOneMethodTrace,
    ThresholdLockArtifact,
    build_deployment_schedule,
    build_pass_one_observation_artifact,
    deployment_runtime_config,
    freeze_deployment_schedule,
    freeze_pass_one_observations,
    freeze_threshold_lock,
    pass_one_trace_from_records,
    threshold_lock_from_calibration,
)
from experiments12.harness12 import ARM_TO_PROBE
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.pairing12 import JobCell, TaskRef, make_pair_manifest
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    effective_passive_method_names,
)
from experiments12.planning_lock12 import (
    ScientificLaunchBinding,
    assert_scientific_launch,
)
from experiments12.runner12 import load_pair_cells, load_task_manifest
from experiments12.signal_integrity12 import (
    SignalIntegrityError,
    validate_active_signal_records,
    validate_passive_signal_records,
)
from experiments12.spec12 import Benchmark, OPERATIONAL_PROVIDER_USD, Operator, Stage
from experiments12.source_registry12 import SourceRegistryError, normalize_source_id
from experiments12.validate12 import validate_run


PREPARATION_VERSION = 1
DEPLOYMENT_MODE = TWO_PASS_DEPLOYMENT_MODE
PASS_ONE_INITIALIZER_VERSION = 1
DEPLOYMENT_PAIR_RECEIPT = "deployment_pair_manifest"
SOURCE_OBSERVATION_MANIFEST_RECEIPT = "source_observation_manifest"
CALIBRATION_MANIFEST_RECEIPT = "calibration_manifest"
CALIBRATION_EXTRACT_RECEIPT = "calibration_analysis_extract"
CALIBRATION_THRESHOLDS_RECEIPT = "calibration_analysis_thresholds"
PASS_ONE_SOURCE_CONTRACT_VERSION = 1
PASS_ONE_SOURCE_CONTRACT_TYPE = "experiment12_deployment_pass_one_source"

_SUPPORTED_OPERATORS = frozenset(
    {
        Operator.NONE.value,
        Operator.COMPACT.value,
        Operator.REGROUND.value,
    }
)


@dataclass(frozen=True, slots=True)
class PreparationResult:
    run_id: str
    layout: RunLayout
    declared_cells: int
    pass_one_traces: int
    pair_manifest_sha256: str
    pass_one_sha256: str
    threshold_lock_sha256: str
    schedule_sha256: str
    manifest_sha256: str


def _regular_file(path: str | Path, label: str) -> Path:
    result = Path(path).resolve()
    if Path(path).is_symlink() or not result.is_file():
        raise FileNotFoundError(f"{label} must be an existing non-symlink file")
    return result


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DeploymentArtifactError(f"{label} must be lowercase SHA256")
    return value


def _unique_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DeploymentArtifactError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(not isinstance(value, str) or not value for value in result)
    ):
        raise DeploymentArtifactError(f"{label} must be nonempty unique names")
    return result


def deployment_pass_one_source_arms(methods: Sequence[str]) -> tuple[str, ...]:
    """Return the only target arms needed to observe ``methods`` once.

    Passive and baseline methods are extracted from the clean trajectory's
    immutable shadow.  Every selected active method needs its own carried
    trajectory.  Keeping this source run smaller than the eventual deployment
    treatment product is both scientifically faithful and materially cheaper.
    """

    method_names = _unique_names(methods, "deployment methods")
    active = tuple(method for method in method_names if method in ARM_TO_PROBE)
    passive = set(effective_passive_method_names())
    if not active or not passive.intersection(method_names):
        raise DeploymentArtifactError(
            "deployment pass one requires at least one active and one passive method"
        )
    unknown = set(method_names) - set(active) - passive
    if unknown:
        raise DeploymentArtifactError(
            f"deployment pass-one methods are unknown: {sorted(unknown)}"
        )
    return ("clean", *active)


def deployment_pass_one_source_contract(
    *,
    methods: Sequence[str],
    operators: Sequence[str],
    estimand: DeploymentEstimand | str,
    natural_max_actions_per_task: int,
    matched_actions_per_method: int,
    yoke_anchor_method: str,
    randomization_seed: int,
    threshold_artifact_sha256: str,
    calibration_manifest_sha256: str,
    planning_lock_sha256: str,
) -> dict[str, Any]:
    """Build the exact manifest contract for a production pass-one run."""

    method_names = _unique_names(methods, "deployment methods")
    operator_names = _unique_names(operators, "deployment operators")
    if (
        set(operator_names) - _SUPPORTED_OPERATORS
        or Operator.NONE.value not in operator_names
        or len(operator_names) < 2
    ):
        raise DeploymentArtifactError(
            "deployment pass one must target none plus compaction/reground operators"
        )
    try:
        policy = (
            estimand
            if isinstance(estimand, DeploymentEstimand)
            else DeploymentEstimand(estimand)
        )
    except (TypeError, ValueError) as exc:
        raise DeploymentArtifactError("deployment pass-one estimand is invalid") from exc
    for name, value in (
        ("natural_max_actions_per_task", natural_max_actions_per_task),
        ("matched_actions_per_method", matched_actions_per_method),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DeploymentArtifactError(f"{name} must be a positive integer")
    if yoke_anchor_method not in method_names:
        raise DeploymentArtifactError("yoke anchor is absent from deployment methods")
    if (
        isinstance(randomization_seed, bool)
        or not isinstance(randomization_seed, int)
        or randomization_seed < 0
    ):
        raise DeploymentArtifactError("deployment randomization seed must be non-negative")
    source_arms = deployment_pass_one_source_arms(method_names)
    return {
        "schema_version": PASS_ONE_SOURCE_CONTRACT_VERSION,
        "artifact_type": PASS_ONE_SOURCE_CONTRACT_TYPE,
        "role": "outcome_blind_deployment_pass_one",
        "statistical_unit": "source_task",
        "replicates": 1,
        "source_arms": list(source_arms),
        "source_operators": [Operator.NONE.value],
        "deployment_methods": list(method_names),
        "deployment_operators": list(operator_names),
        "deployment_estimand": policy.value,
        "natural_max_actions_per_task": natural_max_actions_per_task,
        "matched_actions_per_method": matched_actions_per_method,
        "yoke_anchor_method": yoke_anchor_method,
        "randomization_seed": randomization_seed,
        "threshold_artifact_sha256": _digest(
            threshold_artifact_sha256, "threshold artifact hash"
        ),
        "calibration_manifest_sha256": _digest(
            calibration_manifest_sha256, "calibration manifest hash"
        ),
        "planning_lock_sha256": _digest(
            planning_lock_sha256, "planning lock hash"
        ),
    }


def _require_receipt(
    manifest: Mapping[str, Any], name: str, path: Path
) -> str:
    digest = sha256_file(path)
    receipts = manifest.get("benchmark_receipts")
    if not isinstance(receipts, list):
        raise DeploymentArtifactError("source manifest receipts are invalid")
    matches = [
        receipt
        for receipt in receipts
        if isinstance(receipt, Mapping) and receipt.get("name") == name
    ]
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        raise DeploymentArtifactError(
            f"{name} is absent from or differs from the source manifest"
        )
    return digest


def _require_receipt_sha256(
    manifest: Mapping[str, Any], name: str, expected_sha256: Any
) -> str:
    """Require one receipt by its already-bound digest.

    This is used for upstream artifacts whose original path need not be passed
    to the deployment coordinator.  In particular, the source observation run
    can have its own projection lock while deployment uses a separate lock.
    """

    digest = _digest(expected_sha256, f"{name} hash")
    receipts = manifest.get("benchmark_receipts")
    if not isinstance(receipts, list):
        raise DeploymentArtifactError("source manifest receipts are invalid")
    matches = [
        receipt
        for receipt in receipts
        if isinstance(receipt, Mapping) and receipt.get("name") == name
    ]
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        raise DeploymentArtifactError(
            f"{name} is absent from or differs from its source launch binding"
        )
    return digest


def _validated_run(layout: RunLayout, expected_manifest_sha256: str, label: str) -> None:
    report = validate_run(
        layout,
        repository_root=REPOSITORY_ROOT,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if not report.primary_ready:
        codes = sorted({issue.code for issue in report.errors})
        raise DeploymentArtifactError(
            f"{label} run is not complete and provenance-valid: {', '.join(codes)}"
        )


def _source_task_keys(
    rows: Sequence[Mapping[str, Any]], *, context: str
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        benchmark = row.get("benchmark")
        source_task_id = row.get("source_task_id")
        if (
            not isinstance(benchmark, str)
            or not benchmark
            or not isinstance(source_task_id, str)
            or not source_task_id
        ):
            raise DeploymentArtifactError(
                f"{context} task row {index} lacks canonical benchmark/source_task_id"
            )
        raw_source = source_task_id.split("::", 1)[0]
        try:
            canonical_source = normalize_source_id(benchmark, source_task_id)
        except SourceRegistryError as exc:
            # TurnBench and older provider-free fixtures are outside the source
            # registry. Evolving numeric/prefixed aliases are never allowed to
            # fall back because those are exactly the collision-prone form.
            if (
                benchmark == Benchmark.EVOLVING_GSM8K.value
                and (
                    raw_source.isdecimal()
                    or raw_source.startswith("extracted-gsm8k-test-")
                )
            ):
                raise DeploymentArtifactError(
                    f"{context} task row {index} has invalid Evolving source ID"
                ) from exc
            canonical_source = raw_source
        result.append((benchmark, canonical_source))
    unique = tuple(sorted(set(result)))
    if not unique:
        raise DeploymentArtifactError(f"{context} source-task set is empty")
    return unique


def deployment_threshold_lock_from_analysis(
    threshold_artifact: Mapping[str, Any],
    *,
    deployment_task_rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    methods: Sequence[str],
    natural_max_actions_per_task: int,
    matched_actions_per_method: int,
    yoke_anchor_method: str,
) -> ThresholdLockArtifact:
    """Strictly convert calibration selections after a global split check."""

    model_names = _unique_names(models, "models")
    method_names = _unique_names(methods, "methods")
    try:
        selections, _slices, calibration_sources, required_passive = (
            load_threshold_artifact(threshold_artifact)
        )
    except ValueError as exc:
        raise DeploymentArtifactError(f"invalid analysis threshold artifact: {exc}") from exc
    deployment_sources = set(_source_task_keys(deployment_task_rows, context="deployment"))
    calibration_rows = tuple(
        {"benchmark": benchmark, "source_task_id": source_task_id}
        for benchmark, source_task_id in calibration_sources
    )
    canonical_calibration = _source_task_keys(
        calibration_rows, context="calibration threshold"
    )
    if len(canonical_calibration) != len(calibration_sources):
        raise DeploymentArtifactError(
            "calibration threshold aliases duplicate one canonical source task"
        )
    overlap = deployment_sources.intersection(canonical_calibration)
    if overlap:
        display = ", ".join(
            f"{benchmark}/{source}" for benchmark, source in sorted(overlap)[:5]
        )
        raise DeploymentArtifactError(
            "calibration and deployment source tasks overlap globally: " + display
        )

    passive = set(effective_passive_method_names())
    selected_passive = passive.intersection(method_names)
    selected_active = set(method_names).intersection(ARM_TO_PROBE)
    if not selected_active or not selected_passive:
        raise DeploymentArtifactError(
            "deployment methods must include at least one active and one passive method"
        )
    unknown = set(method_names) - passive - set(ARM_TO_PROBE)
    if unknown:
        raise DeploymentArtifactError(
            f"deployment methods are unknown: {sorted(unknown)}"
        )
    if selected_passive - set(required_passive):
        raise DeploymentArtifactError(
            "selected passive methods are absent from the calibration lock"
        )
    if yoke_anchor_method not in method_names:
        raise DeploymentArtifactError("yoke anchor is absent from deployment methods")

    benchmarks = {str(row["benchmark"]) for row in deployment_task_rows}
    expected = {
        (model, benchmark, method)
        for model in model_names
        for benchmark in benchmarks
        for method in method_names
    }
    by_key = {
        (row.model, row.benchmark, row.method): row for row in selections
    }
    if len(by_key) != len(selections):
        raise DeploymentArtifactError("analysis threshold rows are duplicated")
    missing = expected - set(by_key)
    if missing:
        raise DeploymentArtifactError(
            f"analysis thresholds do not cover deployment slices: {sorted(missing)}"
        )
    selected = tuple(by_key[key] for key in sorted(expected))
    return threshold_lock_from_calibration(
        calibration_run_id=str(threshold_artifact["source_run_id"]),
        calibration_manifest_sha256=str(
            threshold_artifact["source_manifest_sha256"]
        ),
        selections=selected,
        natural_max_actions_per_task=natural_max_actions_per_task,
        matched_actions_per_method=matched_actions_per_method,
        yoke_anchor_method=yoke_anchor_method,
    )


def verify_analysis_threshold_derivation(
    threshold_path: Path,
    extract_path: Path,
) -> tuple[Mapping[str, Any], str]:
    threshold = read_json(threshold_path)
    if not isinstance(threshold, Mapping):
        raise DeploymentArtifactError("analysis threshold artifact must be an object")
    # Strict parsing occurs before any destination is created.
    load_threshold_artifact(threshold)
    extract_sha256 = sha256_file(extract_path)
    if threshold.get("source_extract_sha256") != extract_sha256:
        raise DeploymentArtifactError(
            "calibration extract differs from the analysis threshold provenance"
        )
    extract = read_json(extract_path)
    if not isinstance(extract, Mapping):
        raise DeploymentArtifactError("calibration extract must be an object")
    if (
        extract.get("run_id") != threshold.get("source_run_id")
        or extract.get("manifest_sha256") != threshold.get("source_manifest_sha256")
        or extract.get("stage") != Stage.CALIBRATION.value
        or extract.get("split") != "calibration"
    ):
        raise DeploymentArtifactError(
            "calibration extract identity/stage differs from the threshold artifact"
        )
    raw_traces = extract.get("signal_traces")
    if not isinstance(raw_traces, list) or not raw_traces:
        raise DeploymentArtifactError("calibration extract has no signal traces")
    try:
        traces = tuple(_trace_from_dict(row) for row in raw_traces)
        target = float(threshold["target_firing_rate"])
        selections = calibrate_thresholds(traces, target_firing_rate=target)
        rebuilt = make_threshold_artifact(
            extract,
            traces,
            selections,
            target_firing_rate=target,
            source_extract_sha256=extract_sha256,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentArtifactError(
            f"calibration threshold derivation cannot be reproduced: {exc}"
        ) from exc
    if rebuilt != dict(threshold):
        raise DeploymentArtifactError(
            "analysis thresholds do not reproduce from the frozen calibration extract"
        )
    return threshold, extract_sha256


def verify_calibration_extract_against_run(
    layout: RunLayout,
    extract_path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Re-extract a complete calibration run and require byte-value identity."""

    extract_file = _regular_file(extract_path, "calibration extract")
    try:
        rebuilt = extract_run(
            layout,
            expected_manifest_sha256=_digest(
                expected_manifest_sha256, "calibration manifest hash"
            ),
            split="calibration",
        )
    except ValueError as exc:
        raise DeploymentArtifactError(
            f"calibration run/extract provenance is invalid: {exc}"
        ) from exc
    frozen = read_json(extract_file)
    if rebuilt != frozen:
        raise DeploymentArtifactError(
            "calibration extract does not reproduce from the completed run"
        )
    return frozen


def _record_messages(record: Mapping[str, Any]) -> tuple[Any, ...]:
    native = record.get("messages")
    if native is not None:
        if not isinstance(native, list):
            raise DeploymentArtifactError("native task record messages are invalid")
        return tuple(native)
    return (record.get("user_message"), record.get("assistant_message"))


def _active_records(
    trajectory: Mapping[str, Any], method: str
) -> tuple[Mapping[str, Any], ...]:
    try:
        return validate_active_signal_records(trajectory, method)
    except SignalIntegrityError as exc:
        raise DeploymentArtifactError(f"active signal integrity failed: {exc}") from exc


def _qualified_passive_method(record: Mapping[str, Any]) -> str:
    method = record.get("method")
    variant = record.get("variant")
    if not isinstance(method, str) or not method:
        raise DeploymentArtifactError("passive record has no method")
    if variant is None:
        return method
    if not isinstance(variant, str) or not variant:
        raise DeploymentArtifactError("passive record variant is invalid")
    return f"{method}:{variant}"


def _passive_records(
    trajectory: Mapping[str, Any],
    shadow: Mapping[str, Any],
    method: str,
) -> tuple[Mapping[str, Any], ...]:
    try:
        return validate_passive_signal_records(trajectory, shadow, method)
    except SignalIntegrityError as exc:
        raise DeploymentArtifactError(f"passive signal integrity failed: {exc}") from exc


def _trace_identity(cell: JobCell) -> tuple[str, str, str, str, int]:
    return (
        cell.pair_key.model,
        cell.pair_key.domain,
        cell.pair_key.task_id,
        str(cell.pair_key.task_sha256),
        cell.pair_key.replicate_id,
    )


def _build_pass_one_traces(
    *,
    source_layout: RunLayout,
    source_cells: Sequence[JobCell],
    methods: Sequence[str],
) -> tuple[PassOneMethodTrace, ...]:
    active_methods = set(methods).intersection(ARM_TO_PROBE)
    passive_methods = set(methods).intersection(effective_passive_method_names())
    by_identity_arm: dict[tuple[tuple[str, str, str, str, int], str], JobCell] = {}
    identities: set[tuple[str, str, str, str, int]] = set()
    for cell in source_cells:
        if cell.operator != Operator.NONE.value:
            raise DeploymentArtifactError(
                "source observation run must contain only operator=none cells"
            )
        identity = _trace_identity(cell)
        identities.add(identity)
        key = (identity, cell.arm)
        if key in by_identity_arm:
            raise DeploymentArtifactError("source observation cells are duplicated")
        by_identity_arm[key] = cell

    traces: list[PassOneMethodTrace] = []
    for identity in sorted(identities):
        model, benchmark, task_id, task_sha256, replicate_id = identity
        for method in methods:
            source_arm = method if method in active_methods else "clean"
            try:
                cell = by_identity_arm[(identity, source_arm)]
            except KeyError as exc:
                raise DeploymentArtifactError(
                    f"source run lacks {source_arm} for {identity!r}"
                ) from exc
            trajectory_path = source_layout.trajectories / f"{cell.cell_id}.json"
            trajectory = read_json(trajectory_path)
            horizon = len(trajectory.get("task_records", ()))
            source_trajectory_sha256 = _digest(
                trajectory.get("transcript_sha256"), "source trajectory hash"
            )
            if method in active_methods:
                records = _active_records(trajectory, method)
            elif method in passive_methods:
                shadow_path = source_layout.shadow / f"{cell.cell_id}.json"
                if not shadow_path.is_file() or shadow_path.is_symlink():
                    raise DeploymentArtifactError(
                        f"source run lacks exact clean shadow for {cell.cell_id}"
                    )
                shadow = read_json(shadow_path)
                shadow_events = source_layout.events / f"shadow-{cell.cell_id}.jsonl"
                shadow_job = (
                    source_layout.results / "shadow_jobs" / f"{cell.cell_id}.json"
                )
                if (
                    not shadow_events.is_file()
                    or shadow_events.is_symlink()
                    or not shadow_job.is_file()
                    or shadow_job.is_symlink()
                ):
                    raise DeploymentArtifactError(
                        f"source run lacks shadow event/job receipts for {cell.cell_id}"
                    )
                event_records = read_jsonl(shadow_events)
                if (
                    not isinstance(shadow, Mapping)
                    or event_records != shadow.get("records")
                    or sha256_json(event_records)
                    != sha256_json(shadow.get("records"))
                ):
                    raise DeploymentArtifactError(
                        f"shadow output differs from append-only events for {cell.cell_id}"
                    )
                job = read_json(shadow_job)
                if (
                    not isinstance(job, Mapping)
                    or job.get("cell_id") != cell.cell_id
                    or job.get("state") != "complete"
                    or job.get("shadow_sha256") != sha256_file(shadow_path)
                    or job.get("monitor_methods") != shadow.get("monitor_methods")
                    or job.get("passive_monitor_spec_sha256")
                    != PASSIVE_MONITOR_SPEC_SHA256
                ):
                    raise DeploymentArtifactError(
                        f"shadow output differs from its job/hash receipt for {cell.cell_id}"
                    )
                records = _passive_records(trajectory, shadow, method)
            else:  # guarded by the public threshold helper
                raise DeploymentArtifactError(f"unknown method: {method}")
            traces.append(
                pass_one_trace_from_records(
                    model=model,
                    benchmark=benchmark,
                    task_id=task_id,
                    task_sha256=task_sha256,
                    replicate_id=replicate_id,
                    method=method,
                    source_trajectory_sha256=source_trajectory_sha256,
                    task_horizon=horizon,
                    records=records,
                )
            )
    return tuple(sorted(traces, key=lambda row: row.identity))


def _split_csv(value: str) -> tuple[str, ...]:
    return _unique_names(
        tuple(item.strip() for item in value.split(",") if item.strip()),
        "comma-separated values",
    )


def prepare_deployment_run(
    *,
    source_run_id: str,
    deployment_run_id: str,
    task_manifest_path: str | Path,
    calibration_threshold_path: str | Path,
    calibration_extract_path: str | Path,
    source_registry_path: str | Path,
    baseline_profile_path: str | Path,
    planning_lock_path: str | Path,
    realized_allocation_path: str | Path | None = None,
    methods: Sequence[str],
    operators: Sequence[str],
    estimand: DeploymentEstimand | str,
    natural_max_actions_per_task: int,
    matched_actions_per_method: int,
    yoke_anchor_method: str,
    randomization_seed: int,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS,
    evolving_dataset_path: str | Path | None = None,
    evolving_build_receipt_path: str | Path | None = None,
) -> PreparationResult:
    """Create one complete, immutable deployment preparation chain."""

    method_names = _unique_names(methods, "methods")
    operator_names = _unique_names(operators, "operators")
    if set(operator_names) - _SUPPORTED_OPERATORS:
        raise DeploymentArtifactError(
            "preparation supports none/compaction/reground; feedback needs a separate frozen plan"
        )
    if Operator.NONE.value not in operator_names or len(operator_names) < 2:
        raise DeploymentArtifactError(
            "deployment operators require none plus at least one intervention"
        )
    try:
        policy = (
            estimand
            if isinstance(estimand, DeploymentEstimand)
            else DeploymentEstimand(estimand)
        )
    except (TypeError, ValueError) as exc:
        raise DeploymentArtifactError("deployment estimand is invalid") from exc
    deployment_stage = Stage.CONFIRMATORY
    if (
        isinstance(randomization_seed, bool)
        or not isinstance(randomization_seed, int)
        or randomization_seed < 0
    ):
        raise DeploymentArtifactError("randomization seed must be non-negative")

    artifacts = Path(artifacts_root).resolve()
    source_layout = RunLayout.for_run(artifacts, source_run_id)
    destination = RunLayout.for_run(artifacts, deployment_run_id)
    if destination.root.exists():
        raise FileExistsError("deployment run already exists; preparation is write-once")
    task_file = _regular_file(task_manifest_path, "task manifest")
    threshold_file = _regular_file(
        calibration_threshold_path, "calibration threshold artifact"
    )
    extract_file = _regular_file(
        calibration_extract_path, "calibration extract"
    )
    registry_file = _regular_file(
        source_registry_path, "source allocation registry"
    )
    baseline_profile_file = _regular_file(
        baseline_profile_path, "measured baseline resource profile"
    )
    planning_lock_file = _regular_file(
        planning_lock_path, "cost/sample-size projection lock"
    )
    realized_allocation_file = (
        None
        if realized_allocation_path is None
        else _regular_file(
            realized_allocation_path, "realized source allocation receipt"
        )
    )
    source_manifest_file = _regular_file(
        source_layout.manifest, "source observation manifest"
    )
    source_manifest_sha256 = sha256_file(source_manifest_file)
    source_manifest = read_json(source_manifest_file)
    if not isinstance(source_manifest, Mapping) or source_manifest.get(
        "run_id"
    ) != source_run_id:
        raise DeploymentArtifactError("source observation manifest identity changed")
    if source_manifest.get("stage") != Stage.CONFIRMATORY.value:
        raise DeploymentArtifactError(
            "pass one must come from a confirmatory-cap observation run"
        )
    _require_receipt(source_manifest, "task_manifest", task_file)
    _validated_run(source_layout, source_manifest_sha256, "source observation")

    task_rows = load_task_manifest(task_file)
    if any(
        row["benchmark"] != "evolving_intent_gsm8k" or row["condition"] != "t7"
        for row in task_rows
    ):
        raise DeploymentArtifactError(
            "the current production deployment runner accepts only Evolving Intent t7"
        )
    models = _unique_names(tuple(source_manifest.get("models", ())), "source models")
    source_cells = load_pair_cells(source_layout.pairs)
    source_extra = source_manifest.get("extra_config")
    expected_source_extra_fields = {
        "initializer_version",
        "n_tasks",
        "n_cells",
        "replicates",
        "scientific_launch_lock",
        "analysis_lock",
        "deployment_pass_one_source",
    }
    if (
        not isinstance(source_extra, Mapping)
        or set(source_extra) != expected_source_extra_fields
        or source_extra.get("initializer_version") != PASS_ONE_INITIALIZER_VERSION
        or source_extra.get("n_tasks") != len(task_rows)
        or source_extra.get("n_cells") != len(source_cells)
    ):
        raise DeploymentArtifactError(
            "source observation run lacks exact production initializer metadata"
        )
    replicates = source_extra.get("replicates")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates != 1:
        raise DeploymentArtifactError(
            "deployment pass one requires exactly one replicate per source task"
        )
    if set(source_manifest.get("operators", ())) != {Operator.NONE.value}:
        raise DeploymentArtifactError("source observation operators are not exactly none")
    threshold_metadata = read_json(threshold_file)
    if not isinstance(threshold_metadata, Mapping):
        raise DeploymentArtifactError("calibration threshold artifact is not an object")
    expected_source_contract = deployment_pass_one_source_contract(
        methods=method_names,
        operators=operator_names,
        estimand=policy,
        natural_max_actions_per_task=natural_max_actions_per_task,
        matched_actions_per_method=matched_actions_per_method,
        yoke_anchor_method=yoke_anchor_method,
        randomization_seed=randomization_seed,
        threshold_artifact_sha256=sha256_file(threshold_file),
        calibration_manifest_sha256=_digest(
            threshold_metadata.get("source_manifest_sha256"),
            "threshold calibration manifest hash",
        ),
        planning_lock_sha256=sha256_file(planning_lock_file),
    )
    if (
        source_extra.get("deployment_pass_one_source")
        != expected_source_contract
        or tuple(source_manifest.get("arms", ()))
        != tuple(expected_source_contract["source_arms"])
        or source_manifest.get("randomization_seed") != randomization_seed
    ):
        raise DeploymentArtifactError(
            "source observation run lacks the exact deployment pass-one contract"
        )
    expected_analysis_lock = {
        "threshold_artifact_sha256": sha256_file(threshold_file),
        "calibration_manifest_sha256": expected_source_contract[
            "calibration_manifest_sha256"
        ],
    }
    if source_extra.get("analysis_lock") != expected_analysis_lock:
        raise DeploymentArtifactError(
            "source observation run lacks the exact deployment analysis lock"
        )
    expected_source_identities = {
        (
            model,
            str(row["benchmark"]),
            str(row["task_id"]),
            str(row["task_sha256"]),
            replicate_id,
        )
        for model in models
        for row in task_rows
        for replicate_id in range(replicates)
    }
    actual_source_identities = {_trace_identity(cell) for cell in source_cells}
    if actual_source_identities != expected_source_identities:
        raise DeploymentArtifactError(
            "source pair identities differ from the frozen task/model/replicate product"
        )
    expected_source_treatments = {
        (identity, arm, Operator.NONE.value)
        for identity in expected_source_identities
        for arm in expected_source_contract["source_arms"]
    }
    actual_source_treatments = {
        (_trace_identity(cell), cell.arm, cell.operator) for cell in source_cells
    }
    if (
        actual_source_treatments != expected_source_treatments
        or len(source_cells) != len(expected_source_treatments)
    ):
        raise DeploymentArtifactError(
            "source pair table is not the exact production initializer product"
        )

    try:
        launch_binding = assert_scientific_launch(
            task_rows=task_rows,
            stage=deployment_stage,
            allocation_stage="deployment",
            design_family="deployment",
            models=models,
            arms=method_names,
            operators=operator_names,
            replicates=replicates,
            ledger_path=destination.ledger,
            registry_path=registry_file,
            projection_lock_path=planning_lock_file,
            baseline_profile_path=baseline_profile_file,
            realized_allocation_path=realized_allocation_file,
        )
    except ValueError as exc:
        raise DeploymentArtifactError(f"scientific deployment launch gate failed: {exc}") from exc
    if not isinstance(launch_binding, ScientificLaunchBinding):
        raise DeploymentArtifactError(
            "scientific deployment launch gate returned no projection binding"
        )
    if launch_binding.projection_lock_sha256 != sha256_file(planning_lock_file):
        raise DeploymentArtifactError(
            "scientific deployment launch binding differs from its projection lock"
        )
    source_launch = source_manifest.get("extra_config", {}).get(
        "scientific_launch_lock"
    )
    if (
        not isinstance(source_launch, Mapping)
        or source_launch.get("source_allocation")
        != launch_binding.allocation.as_dict()
    ):
        raise DeploymentArtifactError(
            "source pass-one run is outside the frozen deployment allocation"
        )
    _require_receipt(source_manifest, "source_allocation_registry", registry_file)
    _require_receipt(
        source_manifest, "measured_baseline_resource_profile", baseline_profile_file
    )
    _require_receipt(
        source_manifest, "cost_sample_size_projection_lock", planning_lock_file
    )
    _require_receipt(
        source_manifest, CALIBRATION_THRESHOLDS_RECEIPT, threshold_file
    )
    if source_launch.get("projection_lock_sha256") != sha256_file(planning_lock_file):
        raise DeploymentArtifactError(
            "source pass-one run is not bound to this deployment planning lock"
        )
    if dict(source_launch) != launch_binding.as_dict():
        raise DeploymentArtifactError(
            "source pass-one scientific launch binding differs from final deployment"
        )
    realized_sha256 = launch_binding.allocation.realized_allocation_sha256
    if realized_sha256 is None:
        if realized_allocation_file is not None:
            raise DeploymentArtifactError(
                "realized allocation receipt is absent from the deployment launch binding"
            )
        if any(
            isinstance(receipt, Mapping)
            and receipt.get("name") == "realized_source_allocation"
            for receipt in source_manifest.get("benchmark_receipts", ())
        ):
            raise DeploymentArtifactError(
                "source manifest has an unbound realized allocation receipt"
            )
    else:
        if realized_allocation_file is None:
            raise DeploymentArtifactError(
                "deployment launch binding requires a realized allocation receipt"
            )
        if realized_sha256 != sha256_file(realized_allocation_file):
            raise DeploymentArtifactError(
                "deployment realized allocation differs from its launch binding"
            )
        _require_receipt(
            source_manifest,
            "realized_source_allocation",
            realized_allocation_file,
        )

    threshold_payload, extract_sha256 = verify_analysis_threshold_derivation(
        threshold_file, extract_file
    )
    calibration_run_id = str(threshold_payload["source_run_id"])
    calibration_layout = RunLayout.for_run(artifacts, calibration_run_id)
    calibration_manifest = _regular_file(
        calibration_layout.manifest, "calibration manifest"
    )
    calibration_manifest_sha256 = sha256_file(calibration_manifest)
    if calibration_manifest_sha256 != threshold_payload.get(
        "source_manifest_sha256"
    ):
        raise DeploymentArtifactError(
            "calibration manifest differs from the analysis threshold provenance"
        )
    verify_calibration_extract_against_run(
        calibration_layout,
        extract_file,
        expected_manifest_sha256=calibration_manifest_sha256,
    )

    threshold_lock = deployment_threshold_lock_from_analysis(
        threshold_payload,
        deployment_task_rows=task_rows,
        models=models,
        methods=method_names,
        natural_max_actions_per_task=natural_max_actions_per_task,
        matched_actions_per_method=matched_actions_per_method,
        yoke_anchor_method=yoke_anchor_method,
    )
    traces = _build_pass_one_traces(
        source_layout=source_layout,
        source_cells=source_cells,
        methods=method_names,
    )
    pass_one = build_pass_one_observation_artifact(
        source_run_id=source_run_id,
        source_manifest_sha256=source_manifest_sha256,
        traces=traces,
    )

    task_refs = tuple(
        TaskRef(
            benchmark=str(row["benchmark"]),
            task_id=str(row["task_id"]),
            task_sha256=str(row["task_sha256"]),
        )
        for row in task_rows
    )
    deployment_cells = make_pair_manifest(
        tasks=task_refs,
        models=models,
        arms=method_names,
        operators=operator_names,
        replicates=replicates,
        randomization_seed=randomization_seed,
    )

    evolving_receipts = _evolving_provenance_receipts(
        list(task_rows),
        dataset_path=(
            None if evolving_dataset_path is None else str(evolving_dataset_path)
        ),
        build_receipt_path=(
            None
            if evolving_build_receipt_path is None
            else str(evolving_build_receipt_path)
        ),
    )
    for receipt in evolving_receipts:
        source_matches = [
            row
            for row in source_manifest.get("benchmark_receipts", ())
            if isinstance(row, Mapping) and row.get("name") == receipt.name
        ]
        if len(source_matches) != 1 or source_matches[0].get("sha256") != receipt.sha256:
            raise DeploymentArtifactError(
                f"source observation run used another {receipt.name}"
            )

    # Exercise every schedule invariant before creating the destination. Exact
    # file-byte hashes are substituted only after the immutable files exist.
    build_deployment_schedule(
        estimand=policy,
        cells=deployment_cells,
        pair_manifest_sha256="0" * 64,
        pass_one=pass_one,
        pass_one_artifact_sha256="1" * 64,
        threshold_lock=threshold_lock,
        threshold_lock_sha256="2" * 64,
        feedback_plans={},
    )

    # No destination path exists before every source and split check above.
    destination.create()
    atomic_write_jsonl(
        destination.pairs, [cell.as_dict() for cell in deployment_cells]
    )
    pair_sha256 = sha256_file(destination.pairs)
    pass_one_path = destination.results / "deployment_pass_one.json"
    threshold_path = destination.results / "deployment_threshold_lock.json"
    schedule_path = destination.results / "deployment_schedule.json"
    pass_one_sha256 = freeze_pass_one_observations(pass_one_path, pass_one)
    threshold_sha256 = freeze_threshold_lock(threshold_path, threshold_lock)
    schedule = build_deployment_schedule(
        estimand=policy,
        cells=deployment_cells,
        pair_manifest_sha256=pair_sha256,
        pass_one=pass_one,
        pass_one_artifact_sha256=pass_one_sha256,
        threshold_lock=threshold_lock,
        threshold_lock_sha256=threshold_sha256,
        feedback_plans={},
    )
    schedule_sha256 = freeze_deployment_schedule(
        schedule_path,
        schedule,
        outcome_artifacts_root=destination.results / "deployment",
    )

    receipts = (
        ArtifactReceipt.from_file(
            "task_manifest", task_file, workspace=REPOSITORY_ROOT
        ),
        ArtifactReceipt.from_file(
            DEPLOYMENT_PAIR_RECEIPT,
            destination.pairs,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            SOURCE_OBSERVATION_MANIFEST_RECEIPT,
            source_manifest_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            CALIBRATION_MANIFEST_RECEIPT,
            calibration_manifest,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            CALIBRATION_EXTRACT_RECEIPT,
            extract_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            CALIBRATION_THRESHOLDS_RECEIPT,
            threshold_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            PASS_ONE_RECEIPT, pass_one_path, workspace=REPOSITORY_ROOT
        ),
        ArtifactReceipt.from_file(
            THRESHOLD_LOCK_RECEIPT,
            threshold_path,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            DEPLOYMENT_SCHEDULE_RECEIPT,
            schedule_path,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            "source_allocation_registry",
            registry_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            "measured_baseline_resource_profile",
            baseline_profile_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            "cost_sample_size_projection_lock",
            planning_lock_file,
            workspace=REPOSITORY_ROOT,
        ),
        *(
            ()
            if realized_allocation_file is None
            else (
                ArtifactReceipt.from_file(
                    "realized_source_allocation",
                    realized_allocation_file,
                    workspace=REPOSITORY_ROOT,
                ),
            )
        ),
        *evolving_receipts,
    )
    extra_config: dict[str, Any] = {
        "preparation_version": PREPARATION_VERSION,
        "deployment_mode": DEPLOYMENT_MODE,
        "deployment_estimand": policy.value,
        "n_tasks": len(task_refs),
        "n_cells": len(deployment_cells),
        "replicates": replicates,
        "source_observation_run_id": source_run_id,
        "source_observation_manifest_sha256": source_manifest_sha256,
        "methods": list(method_names),
        "natural_max_actions_per_task": natural_max_actions_per_task,
        "matched_actions_per_method": matched_actions_per_method,
        "yoke_anchor_method": yoke_anchor_method,
        "calibration_extract_sha256": extract_sha256,
        "calibration_threshold_artifact_sha256": sha256_file(threshold_file),
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "deployment_runtime": deployment_runtime_config(),
        "scientific_launch_lock": launch_binding.as_dict(),
    }
    extra_config["analysis_lock"] = {
        "threshold_artifact_sha256": sha256_file(threshold_file),
        "calibration_manifest_sha256": calibration_manifest_sha256,
    }
    manifest = build_manifest(
        run_id=deployment_run_id,
        stage=deployment_stage,
        repository_root=REPOSITORY_ROOT,
        pair_manifest_sha256=pair_sha256,
        models=models,
        arms=method_names,
        operators=operator_names,
        randomization_seed=randomization_seed,
        benchmark_receipts=receipts,
        extra_config=extra_config,
    )
    manifest_sha256 = write_manifest_once(destination.manifest, manifest)
    BudgetLedger(
        destination.ledger,
        operational_caps_usd={
            provider: Decimal(str(amount))
            for provider, amount in OPERATIONAL_PROVIDER_USD.items()
        },
    )
    return PreparationResult(
        run_id=deployment_run_id,
        layout=destination,
        declared_cells=len(deployment_cells),
        pass_one_traces=len(traces),
        pair_manifest_sha256=pair_sha256,
        pass_one_sha256=pass_one_sha256,
        threshold_lock_sha256=threshold_sha256,
        schedule_sha256=schedule_sha256,
        manifest_sha256=manifest_sha256,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--source-run-id", required=True)
    root.add_argument("--deployment-run-id", required=True)
    root.add_argument("--tasks", required=True)
    root.add_argument("--calibration-thresholds", required=True)
    root.add_argument("--calibration-extract", required=True)
    root.add_argument("--source-registry", required=True)
    root.add_argument("--baseline-profile", required=True)
    root.add_argument("--planning-lock", required=True)
    root.add_argument(
        "--realized-allocation",
        help="outcome-blind structural replacement receipt, when needed",
    )
    root.add_argument("--methods", required=True, help="comma-separated active/passive names")
    root.add_argument(
        "--operators",
        default=f"{Operator.NONE.value},{Operator.COMPACT.value},{Operator.REGROUND.value}",
    )
    root.add_argument(
        "--estimand",
        choices=[item.value for item in DeploymentEstimand],
        default=DeploymentEstimand.YOKED_ANCHOR.value,
    )
    root.add_argument("--natural-max-actions-per-task", type=int, default=1)
    root.add_argument("--matched-actions-per-method", type=int, default=1)
    root.add_argument("--yoke-anchor-method", required=True)
    root.add_argument("--seed", type=int, default=120120)
    root.add_argument("--evolving-dataset", required=True)
    root.add_argument("--evolving-build-receipt", required=True)
    root.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = prepare_deployment_run(
            source_run_id=args.source_run_id,
            deployment_run_id=args.deployment_run_id,
            task_manifest_path=args.tasks,
            calibration_threshold_path=args.calibration_thresholds,
            calibration_extract_path=args.calibration_extract,
            source_registry_path=args.source_registry,
            baseline_profile_path=args.baseline_profile,
            planning_lock_path=args.planning_lock,
            realized_allocation_path=args.realized_allocation,
            methods=_split_csv(args.methods),
            operators=_split_csv(args.operators),
            estimand=args.estimand,
            natural_max_actions_per_task=args.natural_max_actions_per_task,
            matched_actions_per_method=args.matched_actions_per_method,
            yoke_anchor_method=args.yoke_anchor_method,
            randomization_seed=args.seed,
            artifacts_root=args.artifacts,
            evolving_dataset_path=args.evolving_dataset,
            evolving_build_receipt_path=args.evolving_build_receipt,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"prepared {result.run_id}: {result.declared_cells} cells, "
        f"{result.pass_one_traces} pass-one traces"
    )
    print(f"manifest={result.layout.manifest}")
    print(f"manifest_sha256={result.manifest_sha256}")
    return 0


__all__ = [
    "CALIBRATION_EXTRACT_RECEIPT",
    "CALIBRATION_MANIFEST_RECEIPT",
    "CALIBRATION_THRESHOLDS_RECEIPT",
    "DEPLOYMENT_MODE",
    "DEPLOYMENT_PAIR_RECEIPT",
    "PREPARATION_VERSION",
    "PreparationResult",
    "SOURCE_OBSERVATION_MANIFEST_RECEIPT",
    "deployment_threshold_lock_from_analysis",
    "main",
    "prepare_deployment_run",
    "verify_analysis_threshold_derivation",
    "verify_calibration_extract_against_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
