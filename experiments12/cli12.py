"""Safe command line entry points for planning and auditing Experiment 12."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import unittest

from experiments12.core.artifacts import (
    atomic_write_jsonl,
    read_json,
    sha256_file,
    sha256_json,
)
from experiments12.core.budget import BudgetLedger
from experiments12.domains.evolving_intent import PINNED_COMMIT as EVOLVING_PINNED_COMMIT
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    validate_manifest_files,
    write_manifest_once,
)
from experiments12.models12 import TARGET_MODEL_NAMES, preflight_model_availability
from experiments12.passive_spec12 import effective_passive_method_names
from experiments12.pairing12 import TaskRef, make_pair_manifest
from experiments12.planning_lock12 import (
    ScientificLaunchBinding,
    assert_scientific_launch,
)
from experiments12.source_registry12 import SourceAllocationBinding
from experiments12.spec12 import (
    ARMS,
    OPERATIONAL_PROVIDER_USD,
    ObservationKind,
    Stage,
    arm as get_arm,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_ARTIFACTS = PACKAGE_ROOT / "artifacts"


def _dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without interpolation or output."""

    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or "=" not in line:
            raise ValueError(f"unsupported .env syntax at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key.replace("_", "").isalnum() or not key[0].isalpha():
            raise ValueError(f"invalid .env key at line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _environment(env_file: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    if env_file:
        environment.update(_dotenv(Path(env_file)))
    return environment


def _load_task_rows(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    if not isinstance(rows, list) or not rows:
        raise ValueError("task manifest must be a nonempty JSON/JSONL list")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("each task row must be an object")
    return rows


def _load_tasks(path: Path) -> tuple[TaskRef, ...]:
    rows = _load_task_rows(path)
    tasks = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"task row {index} must be an object")
        benchmark = row.get("benchmark")
        task_id = row.get("task_id")
        if not isinstance(benchmark, str) or not isinstance(task_id, str):
            raise ValueError(f"task row {index} requires benchmark/task_id strings")
        task_hash = row.get("task_sha256")
        if task_hash is None:
            task_hash = sha256_json(row)
        tasks.append(TaskRef(benchmark, task_id, task_hash))
    return tuple(tasks)


def _reject_active_t1_rows(rows: list[dict[str, object]], arms: tuple[str, ...]) -> None:
    try:
        active = tuple(
            name for name in arms if get_arm(name).observation is ObservationKind.ACTIVE
        )
    except KeyError as exc:
        raise ValueError(f"unknown observation arm: {exc.args[0]}") from exc
    if not active:
        return
    for row in rows:
        condition = row.get("condition")
        task_id = row.get("task_id")
        if condition == "t1" or (
            isinstance(task_id, str) and task_id.rsplit("::", 1)[-1] == "t1"
        ):
            raise ValueError(
                "active observation arms are forbidden for t1 tasks; "
                "initialize the t1 clean baseline separately"
            )


def _evolving_provenance_receipts(
    rows: list[dict[str, object]],
    *,
    dataset_path: str | None,
    build_receipt_path: str | None,
) -> tuple[ArtifactReceipt, ...]:
    """Bind the rendered dataset and the receipt that generated it."""

    evolving_rows = [
        row for row in rows if row.get("benchmark") == "evolving_intent_gsm8k"
    ]
    if not evolving_rows:
        if dataset_path is not None or build_receipt_path is not None:
            raise ValueError("Evolving provenance was supplied for a non-Evolving task run")
        return ()
    if dataset_path is None or build_receipt_path is None:
        raise ValueError(
            "Evolving runs require --evolving-dataset and --evolving-build-receipt"
        )
    dataset = Path(dataset_path).resolve()
    build_receipt = Path(build_receipt_path).resolve()
    dataset_sha = sha256_file(dataset)
    source_hashes = {row.get("source_sha256") for row in evolving_rows}
    if source_hashes != {dataset_sha}:
        raise ValueError("Evolving task manifest is not derived from the supplied dataset")
    payload = read_json(build_receipt)
    frozen_dataset = payload.get("frozen_dataset")
    if (
        payload.get("benchmark") != "evolving_intent_gsm8k"
        or payload.get("upstream_commit") != EVOLVING_PINNED_COMMIT
        or payload.get("shared_across_target_arms_and_models") is not True
        or not isinstance(frozen_dataset, dict)
        or frozen_dataset.get("sha256") != dataset_sha
    ):
        raise ValueError("Evolving build receipt does not attest the supplied dataset")
    return (
        ArtifactReceipt.from_file(
            "evolving_rendered_dataset",
            dataset,
            workspace=REPOSITORY_ROOT,
            upstream_commit=EVOLVING_PINNED_COMMIT,
            license_id="MIT",
        ),
        ArtifactReceipt.from_file(
            "evolving_build_receipt",
            build_receipt,
            workspace=REPOSITORY_ROOT,
            upstream_commit=EVOLVING_PINNED_COMMIT,
            license_id="MIT",
        ),
    )


def _split(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("comma-separated values must be nonempty and unique")
    return result


def _confirmatory_analysis_lock(
    *,
    stage: Stage,
    task_rows: Sequence[Mapping[str, Any]],
    thresholds_path: str | Path | None,
) -> dict[str, str] | None:
    """Validate and bind the calibration artifact before a test manifest exists."""

    if stage is not Stage.CONFIRMATORY:
        if thresholds_path is not None:
            raise ValueError("--thresholds is permitted only for confirmatory init")
        return None
    if thresholds_path is None:
        raise ValueError(
            "confirmatory init requires --thresholds so its hash can be frozen"
        )
    threshold_path = Path(thresholds_path).resolve()
    locked = read_json(threshold_path)
    if (
        not isinstance(locked, dict)
        or locked.get("artifact_type") != "locked_fixed_rate_thresholds"
    ):
        raise ValueError("--thresholds is not an Experiment 12 lock artifact")
    calibration_sha = locked.get("source_manifest_sha256")
    if (
        not isinstance(calibration_sha, str)
        or len(calibration_sha) != 64
        or any(character not in "0123456789abcdef" for character in calibration_sha)
    ):
        raise ValueError("threshold artifact lacks a calibration manifest SHA256")
    frozen_passive = locked.get("required_passive_methods")
    required_passive = effective_passive_method_names()
    if (
        not isinstance(frozen_passive, list)
        or not frozen_passive
        or frozen_passive != sorted(set(frozen_passive))
        or any(not isinstance(item, str) or not item for item in frozen_passive)
    ):
        raise ValueError("threshold artifact lacks a frozen passive method set")
    if tuple(frozen_passive) != required_passive:
        raise ValueError(
            "threshold passive methods differ from the canonical passive spec"
        )
    calibration_rows = locked.get("calibration_source_tasks")
    if not isinstance(calibration_rows, list) or not calibration_rows:
        raise ValueError("threshold artifact lacks calibration source tasks")
    calibration_sources: set[tuple[str, str]] = set()
    for row in calibration_rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"benchmark", "source_task_id"}
            or not isinstance(row["benchmark"], str)
            or not row["benchmark"]
            or not isinstance(row["source_task_id"], str)
            or not row["source_task_id"]
        ):
            raise ValueError("threshold calibration source-task row is invalid")
        calibration_sources.add((row["benchmark"], row["source_task_id"]))
    if len(calibration_sources) != len(calibration_rows):
        raise ValueError("threshold calibration source tasks are duplicated")
    confirmatory_sources: set[tuple[str, str]] = set()
    for row in task_rows:
        benchmark, pair_task_id = row.get("benchmark"), row.get("task_id")
        source_task_id = row.get("source_task_id")
        if source_task_id is None and isinstance(pair_task_id, str):
            parts = pair_task_id.split("::")
            source_task_id = parts[0] if len(parts) == 2 else None
        if (
            not isinstance(benchmark, str)
            or not benchmark
            or not isinstance(source_task_id, str)
            or not source_task_id
        ):
            raise ValueError(
                "confirmatory task rows require canonical source-task identities"
            )
        confirmatory_sources.add((benchmark, source_task_id))
    calibration_benchmarks = {benchmark for benchmark, _ in calibration_sources}
    confirmatory_benchmarks = {benchmark for benchmark, _ in confirmatory_sources}
    if len(confirmatory_benchmarks) != 1 or calibration_benchmarks != confirmatory_benchmarks:
        raise ValueError(
            "threshold calibration benchmark differs from the confirmatory task benchmark"
        )
    overlap = calibration_sources.intersection(confirmatory_sources)
    if overlap:
        display = ", ".join(
            f"{benchmark}/{source}" for benchmark, source in sorted(overlap)[:5]
        )
        raise ValueError(
            "calibration and confirmatory source tasks overlap globally: " + display
        )
    return {
        "threshold_artifact_sha256": sha256_file(threshold_path),
        "calibration_manifest_sha256": calibration_sha,
    }


def cmd_preflight(args: argparse.Namespace) -> int:
    results = preflight_model_availability(environ=_environment(args.env_file))
    for result in results:
        flag = "OK" if result.available is True else "NO" if result.available is False else "?"
        print(f"{flag:>2}  {result.provider:<9} {result.name:<24} {result.status}")
    return 0 if all(result.available is True for result in results) else 2


def cmd_init(args: argparse.Namespace) -> int:
    stage = Stage(args.stage)
    tasks_path = Path(args.tasks).resolve()
    task_rows = _load_task_rows(tasks_path)
    tasks = _load_tasks(tasks_path)
    models = TARGET_MODEL_NAMES if args.models == "default" else _split(args.models)
    if args.arms == "core":
        arms = tuple(item.name for item in ARMS if item.confirmatory_core)
    elif args.arms == "pilot":
        arms = tuple(item.name for item in ARMS)
    else:
        arms = _split(args.arms)
    _reject_active_t1_rows(task_rows, arms)
    operators = _split(args.operators)
    layout = RunLayout.for_run(args.artifacts, args.run_id)
    launch_binding = assert_scientific_launch(
        task_rows=task_rows,
        stage=stage,
        models=models,
        arms=arms,
        operators=operators,
        replicates=args.replicates,
        ledger_path=layout.ledger,
        registry_path=args.source_registry,
        projection_lock_path=args.planning_lock,
        baseline_profile_path=args.baseline_profile,
        smoke_wave=args.smoke_wave,
        realized_allocation_path=args.realized_allocation,
    )
    cells = make_pair_manifest(
        tasks=tasks,
        models=models,
        arms=arms,
        operators=operators,
        replicates=args.replicates,
        randomization_seed=args.seed,
    )
    required_passive = effective_passive_method_names()
    if args.required_passive_methods is not None:
        requested_passive = tuple(sorted(_split(args.required_passive_methods)))
        if requested_passive != required_passive:
            raise ValueError(
                "--required-passive-methods differs from the canonical passive spec"
            )
    analysis_lock = _confirmatory_analysis_lock(
        stage=stage,
        task_rows=task_rows,
        thresholds_path=args.thresholds,
    )
    extra_config = {
        "replicates": args.replicates,
        "n_tasks": len(tasks),
        "n_cells": len(cells),
    }
    if isinstance(launch_binding, ScientificLaunchBinding):
        extra_config["scientific_launch_lock"] = launch_binding.as_dict()
    elif isinstance(launch_binding, SourceAllocationBinding):
        extra_config["source_allocation"] = launch_binding.as_dict()
    if analysis_lock is not None:
        extra_config["analysis_lock"] = analysis_lock
    receipt = ArtifactReceipt.from_file(
        "task_manifest",
        tasks_path,
        workspace=REPOSITORY_ROOT,
        license_id=args.task_license,
    )
    evolving_receipts = _evolving_provenance_receipts(
        task_rows,
        dataset_path=getattr(args, "evolving_dataset", None),
        build_receipt_path=getattr(args, "evolving_build_receipt", None),
    )
    launch_receipts: list[ArtifactReceipt] = []
    if args.source_registry is not None:
        launch_receipts.append(
            ArtifactReceipt.from_file(
                "source_allocation_registry",
                args.source_registry,
                workspace=REPOSITORY_ROOT,
            )
        )
    if args.realized_allocation is not None:
        launch_receipts.append(
            ArtifactReceipt.from_file(
                "realized_source_allocation",
                args.realized_allocation,
                workspace=REPOSITORY_ROOT,
            )
        )
    if args.baseline_profile is not None:
        launch_receipts.append(
            ArtifactReceipt.from_file(
                "measured_baseline_resource_profile",
                args.baseline_profile,
                workspace=REPOSITORY_ROOT,
            )
        )
    if args.planning_lock is not None:
        launch_receipts.append(
            ArtifactReceipt.from_file(
                "cost_sample_size_projection_lock",
                args.planning_lock,
                workspace=REPOSITORY_ROOT,
            )
        )
    if layout.pairs.exists() or layout.manifest.exists():
        raise FileExistsError("run already initialized; choose a new run_id")
    layout.create()
    atomic_write_jsonl(layout.pairs, [cell.as_dict() for cell in cells])
    manifest = build_manifest(
        run_id=args.run_id,
        stage=stage,
        repository_root=REPOSITORY_ROOT,
        pair_manifest_sha256=sha256_file(layout.pairs),
        models=models,
        arms=arms,
        operators=operators,
        randomization_seed=args.seed,
        benchmark_receipts=(receipt, *evolving_receipts, *launch_receipts),
        extra_config=extra_config,
    )
    write_manifest_once(layout.manifest, manifest)
    BudgetLedger(
        layout.ledger,
        operational_caps_usd={
            provider: Decimal(str(amount))
            for provider, amount in OPERATIONAL_PROVIDER_USD.items()
        },
    )
    print(f"initialized {args.run_id}: {len(tasks)} tasks, {len(cells)} locked cells")
    print(f"manifest: {layout.manifest}")
    print(f"global budget ledger: {layout.ledger}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    layout = RunLayout.for_run(args.artifacts, args.run_id)
    if not layout.manifest.exists() or not layout.pairs.exists():
        raise FileNotFoundError("run is not initialized")
    manifest = read_json(layout.manifest)
    errors = validate_manifest_files(
        manifest,
        repository_root=REPOSITORY_ROOT,
        pair_manifest_path=layout.pairs,
    )
    print(f"run={args.run_id} stage={manifest['stage']} locked_cells={manifest['extra_config']['n_cells']}")
    print("manifest=" + ("VALID" if not errors else "INVALID: " + "; ".join(errors)))
    ledger = BudgetLedger(layout.ledger)
    for provider, budget in ledger.snapshot().items():
        print(
            f"{provider}: spent=${budget.spent_usd} reserved=${budget.reserved_usd} "
            f"operational_remaining=${budget.remaining_operational_usd} "
            f"hard_remaining=${budget.remaining_hard_usd}"
        )
    return 0 if not errors else 3


def cmd_selftest(_: argparse.Namespace) -> int:
    suite = unittest.defaultTestLoader.discover(str(PACKAGE_ROOT), pattern="test*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="free provider /models check")
    preflight.add_argument("--env-file", default=str(REPOSITORY_ROOT / ".env"))
    preflight.set_defaults(func=cmd_preflight)

    init = commands.add_parser("init", help="freeze a run and its full pair manifest")
    init.add_argument("--run-id", required=True)
    init.add_argument("--stage", choices=[stage.value for stage in Stage], required=True)
    init.add_argument("--tasks", required=True)
    init.add_argument("--task-license", default=None)
    init.add_argument("--evolving-dataset", default=None)
    init.add_argument("--evolving-build-receipt", default=None)
    init.add_argument(
        "--source-registry",
        default=None,
        help="tracked source allocation; required for baseline/calibration/confirmatory",
    )
    init.add_argument(
        "--baseline-profile",
        default=None,
        help="measured clean-baseline resource profile; required after baseline_gate",
    )
    init.add_argument(
        "--planning-lock",
        default=None,
        help="cost/sample-size projection lock; required for calibration/confirmatory",
    )
    init.add_argument(
        "--smoke-wave",
        choices=("single_model", "all_models"),
        default=None,
        help="optional exact smoke subset when --source-registry is supplied",
    )
    init.add_argument(
        "--realized-allocation",
        default=None,
        help="hashed outcome-blind structural replacement receipt, when needed",
    )
    init.add_argument("--models", default="default")
    init.add_argument("--arms", default="pilot", help="pilot, core, or comma list")
    init.add_argument("--operators", default="none")
    init.add_argument("--replicates", type=int, default=1)
    init.add_argument("--seed", type=int, default=120120)
    init.add_argument(
        "--required-passive-methods",
        default=None,
        help="comma list of exact effective passive names, including variants",
    )
    init.add_argument(
        "--thresholds",
        default=None,
        help="locked calibration artifact; required for confirmatory init",
    )
    init.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    init.set_defaults(func=cmd_init)

    status = commands.add_parser("status", help="verify frozen hashes and budget")
    status.add_argument("--run-id", required=True)
    status.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    status.set_defaults(func=cmd_status)

    selftest = commands.add_parser("selftest", help="run all offline Experiment 12 tests")
    selftest.set_defaults(func=cmd_selftest)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
