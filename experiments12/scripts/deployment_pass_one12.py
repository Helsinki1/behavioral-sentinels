"""Provider-free initializer for a two-pass deployment observation run.

The final deployment planning lock describes the methods and recovery
operators that will be compared.  Pass one must not execute that whole product:
it needs one clean trajectory (for every passive/baseline shadow) plus one
trajectory for each carried active method, all under ``operator=none``.  This
module freezes that smaller source run while binding it to the final deployment
allocation, thresholds, measured profile, and cost lock.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments12.analysis12 import load_threshold_artifact
from experiments12.cli12 import (
    DEFAULT_ARTIFACTS,
    REPOSITORY_ROOT,
    _evolving_provenance_receipts,
)
from experiments12.core.artifacts import (
    atomic_write_jsonl,
    read_json,
    sha256_file,
)
from experiments12.core.budget import BudgetLedger
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.pairing12 import TaskRef, make_pair_manifest
from experiments12.planning_lock12 import ScientificLaunchBinding, assert_scientific_launch
from experiments12.prepare_deployment12 import (
    CALIBRATION_THRESHOLDS_RECEIPT,
    PASS_ONE_INITIALIZER_VERSION,
    DeploymentArtifactError,
    DeploymentEstimand,
    deployment_pass_one_source_arms,
    deployment_pass_one_source_contract,
    deployment_threshold_lock_from_analysis,
)
from experiments12.runner12 import load_task_manifest
from experiments12.spec12 import OPERATIONAL_PROVIDER_USD, Operator, Stage


@dataclass(frozen=True, slots=True)
class PassOneInitializationResult:
    run_id: str
    manifest_path: Path
    manifest_sha256: str
    pair_manifest_path: Path
    pair_manifest_sha256: str
    declared_cells: int
    source_arms: tuple[str, ...]


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve()
    if candidate.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"{label} must be an existing non-symlink file")
    return resolved


def _unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DeploymentArtifactError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(not isinstance(item, str) or not item for item in result)
    ):
        raise DeploymentArtifactError(f"{label} must contain unique nonempty names")
    return result


def _split_csv(value: str) -> tuple[str, ...]:
    return _unique(
        tuple(item.strip() for item in value.split(",") if item.strip()),
        "comma-separated values",
    )


def initialize_evolving_pass_one(
    *,
    run_id: str,
    task_manifest_path: str | Path,
    calibration_threshold_path: str | Path,
    source_registry_path: str | Path,
    baseline_profile_path: str | Path,
    planning_lock_path: str | Path,
    models: Sequence[str],
    methods: Sequence[str],
    deployment_operators: Sequence[str],
    estimand: DeploymentEstimand | str,
    natural_max_actions_per_task: int,
    matched_actions_per_method: int,
    yoke_anchor_method: str,
    randomization_seed: int,
    evolving_dataset_path: str | Path,
    evolving_build_receipt_path: str | Path,
    realized_allocation_path: str | Path | None = None,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS,
) -> PassOneInitializationResult:
    """Freeze the exact clean/active observation source for two-pass replay."""

    model_names = _unique(models, "models")
    method_names = _unique(methods, "methods")
    operator_names = _unique(deployment_operators, "deployment operators")
    if (
        isinstance(randomization_seed, bool)
        or not isinstance(randomization_seed, int)
        or randomization_seed < 0
    ):
        raise DeploymentArtifactError("randomization seed must be non-negative")

    task_file = _regular_file(task_manifest_path, "task manifest")
    threshold_file = _regular_file(
        calibration_threshold_path, "calibration threshold artifact"
    )
    registry_file = _regular_file(source_registry_path, "source allocation registry")
    profile_file = _regular_file(
        baseline_profile_path, "measured baseline resource profile"
    )
    lock_file = _regular_file(planning_lock_path, "deployment planning lock")
    dataset_file = _regular_file(evolving_dataset_path, "Evolving rendered dataset")
    build_file = _regular_file(
        evolving_build_receipt_path, "Evolving build receipt"
    )
    realized_file = (
        None
        if realized_allocation_path is None
        else _regular_file(realized_allocation_path, "realized source allocation")
    )

    task_rows = load_task_manifest(task_file)
    if any(
        row.get("benchmark") != "evolving_intent_gsm8k"
        or row.get("condition") != "t7"
        or isinstance(row.get("num_turns"), bool)
        or not isinstance(row.get("num_turns"), int)
        or row["num_turns"] < 2
        for row in task_rows
    ):
        raise DeploymentArtifactError(
            "deployment pass one requires only multi-turn Evolving Intent t7 tasks"
        )

    threshold_payload = read_json(threshold_file)
    if not isinstance(threshold_payload, Mapping):
        raise DeploymentArtifactError("calibration threshold artifact is not an object")
    try:
        load_threshold_artifact(threshold_payload)
    except ValueError as exc:
        raise DeploymentArtifactError(f"invalid calibration threshold artifact: {exc}") from exc
    # This shared conversion validates active/passive coverage, exact selected
    # model/method slices, and global calibration/deployment source disjointness.
    threshold_lock = deployment_threshold_lock_from_analysis(
        threshold_payload,
        deployment_task_rows=task_rows,
        models=model_names,
        methods=method_names,
        natural_max_actions_per_task=natural_max_actions_per_task,
        matched_actions_per_method=matched_actions_per_method,
        yoke_anchor_method=yoke_anchor_method,
    )
    del threshold_lock

    source_arms = deployment_pass_one_source_arms(method_names)
    layout = RunLayout.for_run(artifacts_root, run_id)
    if layout.root.exists():
        raise FileExistsError("deployment pass-one run already exists; choose a new run_id")
    launch = assert_scientific_launch(
        task_rows=task_rows,
        stage=Stage.CONFIRMATORY,
        allocation_stage="deployment",
        design_family="deployment",
        models=model_names,
        arms=method_names,
        operators=operator_names,
        replicates=1,
        ledger_path=layout.ledger,
        registry_path=registry_file,
        projection_lock_path=lock_file,
        baseline_profile_path=profile_file,
        realized_allocation_path=realized_file,
    )
    if not isinstance(launch, ScientificLaunchBinding):
        raise DeploymentArtifactError(
            "deployment pass-one scientific launch gate returned no binding"
        )

    task_refs = tuple(
        TaskRef(
            str(row["benchmark"]), str(row["task_id"]), str(row["task_sha256"])
        )
        for row in task_rows
    )
    cells = make_pair_manifest(
        tasks=task_refs,
        models=model_names,
        arms=source_arms,
        operators=(Operator.NONE.value,),
        replicates=1,
        randomization_seed=randomization_seed,
    )
    evolving_receipts = _evolving_provenance_receipts(
        task_rows,
        dataset_path=str(dataset_file),
        build_receipt_path=str(build_file),
    )
    calibration_manifest_sha256 = threshold_payload.get("source_manifest_sha256")
    contract = deployment_pass_one_source_contract(
        methods=method_names,
        operators=operator_names,
        estimand=estimand,
        natural_max_actions_per_task=natural_max_actions_per_task,
        matched_actions_per_method=matched_actions_per_method,
        yoke_anchor_method=yoke_anchor_method,
        randomization_seed=randomization_seed,
        threshold_artifact_sha256=sha256_file(threshold_file),
        calibration_manifest_sha256=str(calibration_manifest_sha256),
        planning_lock_sha256=sha256_file(lock_file),
    )
    receipts = (
        ArtifactReceipt.from_file("task_manifest", task_file, workspace=REPOSITORY_ROOT),
        *evolving_receipts,
        ArtifactReceipt.from_file(
            CALIBRATION_THRESHOLDS_RECEIPT,
            threshold_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            "source_allocation_registry", registry_file, workspace=REPOSITORY_ROOT
        ),
        ArtifactReceipt.from_file(
            "measured_baseline_resource_profile",
            profile_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            "cost_sample_size_projection_lock", lock_file, workspace=REPOSITORY_ROOT
        ),
        *(
            ()
            if realized_file is None
            else (
                ArtifactReceipt.from_file(
                    "realized_source_allocation",
                    realized_file,
                    workspace=REPOSITORY_ROOT,
                ),
            )
        ),
    )
    extra_config: dict[str, Any] = {
        "initializer_version": PASS_ONE_INITIALIZER_VERSION,
        "n_tasks": len(task_rows),
        "n_cells": len(cells),
        "replicates": 1,
        "scientific_launch_lock": launch.as_dict(),
        "analysis_lock": {
            "threshold_artifact_sha256": sha256_file(threshold_file),
            "calibration_manifest_sha256": calibration_manifest_sha256,
        },
        "deployment_pass_one_source": contract,
    }

    # Every external validation above precedes the write-once run namespace.
    layout.create()
    atomic_write_jsonl(layout.pairs, [cell.as_dict() for cell in cells])
    pair_sha256 = sha256_file(layout.pairs)
    manifest = build_manifest(
        run_id=run_id,
        stage=Stage.CONFIRMATORY,
        repository_root=REPOSITORY_ROOT,
        pair_manifest_sha256=pair_sha256,
        models=model_names,
        arms=source_arms,
        operators=(Operator.NONE.value,),
        randomization_seed=randomization_seed,
        benchmark_receipts=receipts,
        extra_config=extra_config,
    )
    manifest_sha256 = write_manifest_once(layout.manifest, manifest)
    BudgetLedger(
        layout.ledger,
        operational_caps_usd={
            provider: Decimal(str(amount))
            for provider, amount in OPERATIONAL_PROVIDER_USD.items()
        },
    )
    return PassOneInitializationResult(
        run_id=run_id,
        manifest_path=layout.manifest,
        manifest_sha256=manifest_sha256,
        pair_manifest_path=layout.pairs,
        pair_manifest_sha256=pair_sha256,
        declared_cells=len(cells),
        source_arms=source_arms,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--run-id", required=True)
    root.add_argument("--tasks", required=True)
    root.add_argument("--calibration-thresholds", required=True)
    root.add_argument("--source-registry", required=True)
    root.add_argument("--baseline-profile", required=True)
    root.add_argument("--planning-lock", required=True)
    root.add_argument("--realized-allocation")
    root.add_argument("--models", required=True)
    root.add_argument("--methods", required=True)
    root.add_argument("--deployment-operators", required=True)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = initialize_evolving_pass_one(
            run_id=args.run_id,
            task_manifest_path=args.tasks,
            calibration_threshold_path=args.calibration_thresholds,
            source_registry_path=args.source_registry,
            baseline_profile_path=args.baseline_profile,
            planning_lock_path=args.planning_lock,
            realized_allocation_path=args.realized_allocation,
            models=_split_csv(args.models),
            methods=_split_csv(args.methods),
            deployment_operators=_split_csv(args.deployment_operators),
            estimand=args.estimand,
            natural_max_actions_per_task=args.natural_max_actions_per_task,
            matched_actions_per_method=args.matched_actions_per_method,
            yoke_anchor_method=args.yoke_anchor_method,
            randomization_seed=args.seed,
            evolving_dataset_path=args.evolving_dataset,
            evolving_build_receipt_path=args.evolving_build_receipt,
            artifacts_root=args.artifacts,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        f"initialized deployment pass-one run={result.run_id} "
        f"cells={result.declared_cells} manifest={result.manifest_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PASS_ONE_INITIALIZER_VERSION",
    "PassOneInitializationResult",
    "initialize_evolving_pass_one",
    "main",
    "parser",
]
