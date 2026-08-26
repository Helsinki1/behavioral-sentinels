"""Generate and validate Experiment 12 baseline profiles and planning locks.

This command performs local artifact and ledger checks only.  It never reads
provider credentials and never dispatches a model request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from experiments12.core.artifacts import read_json, sha256_file
from experiments12.manifest12 import RunLayout
from experiments12.planning_lock12 import (
    DEPLOYMENT_DESIGN,
    OBSERVER_EFFECT_DESIGN,
    build_baseline_resource_profile,
    build_projection_lock,
    freeze_baseline_resource_profile,
    freeze_projection_lock,
    validate_baseline_resource_profile,
    validate_projection_lock_current_budget,
)
from experiments12.source_registry12 import SOURCE_REGISTRY_PATH
from experiments12.spec12 import Benchmark, Stage


DEFAULT_ARTIFACTS = Path(__file__).with_name("artifacts")


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("comma list must contain unique nonempty values")
    return result


def _profile(args: argparse.Namespace) -> dict[str, str]:
    layouts = tuple(
        RunLayout.for_run(args.artifacts_root, run_id) for run_id in args.run_id
    )
    profile = build_baseline_resource_profile(
        layouts,
        registry_path=args.registry,
    )
    digest = freeze_baseline_resource_profile(
        args.output,
        profile,
        registry_path=args.registry,
    )
    return {"artifact": str(args.output), "sha256": digest, "status": "frozen"}


def _lock(args: argparse.Namespace) -> dict[str, str]:
    lock = build_projection_lock(
        baseline_profile_path=args.baseline_profile,
        registry_path=args.registry,
        ledger_path=args.ledger,
        stage=Stage(args.stage),
        allocation_stage=args.allocation_stage,
        design_family=args.design_family,
        benchmark=args.benchmark,
        models=_csv(args.models),
        arms=_csv(args.arms),
        operators=_csv(args.operators),
        replicates=args.replicates,
        realized_allocation_path=args.realized_allocation,
    )
    digest = freeze_projection_lock(
        args.output,
        lock,
        baseline_profile_path=args.baseline_profile,
        registry_path=args.registry,
        realized_allocation_path=args.realized_allocation,
    )
    return {"artifact": str(args.output), "sha256": digest, "status": "frozen"}


def _validate_profile(args: argparse.Namespace) -> dict[str, str]:
    validate_baseline_resource_profile(
        read_json(args.profile), registry_path=args.registry
    )
    return {
        "artifact": str(args.profile),
        "sha256": sha256_file(args.profile),
        "status": "valid",
    }


def _validate_lock(args: argparse.Namespace) -> dict[str, str]:
    validate_projection_lock_current_budget(
        read_json(args.lock),
        baseline_profile_path=args.baseline_profile,
        registry_path=args.registry,
        ledger_path=args.ledger,
        realized_allocation_path=args.realized_allocation,
    )
    return {
        "artifact": str(args.lock),
        "sha256": sha256_file(args.lock),
        "status": "valid_and_within_current_budget",
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    profile = commands.add_parser(
        "profile",
        help="freeze clean-baseline success and p95 resource profiles",
    )
    profile.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    profile.add_argument("--run-id", action="append", required=True)
    profile.add_argument("--registry", type=Path, default=SOURCE_REGISTRY_PATH)
    profile.add_argument("--output", type=Path, required=True)
    profile.set_defaults(handler=_profile)

    lock = commands.add_parser(
        "lock", help="freeze a sample-size and provider-cost projection"
    )
    lock.add_argument("--baseline-profile", type=Path, required=True)
    lock.add_argument("--registry", type=Path, default=SOURCE_REGISTRY_PATH)
    lock.add_argument("--ledger", type=Path, required=True)
    lock.add_argument(
        "--stage",
        choices=(Stage.CALIBRATION.value, Stage.CONFIRMATORY.value),
        required=True,
    )
    lock.add_argument("--allocation-stage")
    lock.add_argument(
        "--realized-allocation",
        type=Path,
        help="outcome-blind structural replacement receipt, when needed",
    )
    lock.add_argument(
        "--design-family",
        choices=(OBSERVER_EFFECT_DESIGN, DEPLOYMENT_DESIGN),
        default=OBSERVER_EFFECT_DESIGN,
    )
    lock.add_argument(
        "--benchmark",
        choices=(Benchmark.EVOLVING_GSM8K.value, Benchmark.BFCL_MULTI_TURN.value),
        required=True,
    )
    lock.add_argument("--models", required=True)
    lock.add_argument("--arms", required=True)
    lock.add_argument("--operators", required=True)
    lock.add_argument("--replicates", type=int, default=1)
    lock.add_argument("--output", type=Path, required=True)
    lock.set_defaults(handler=_lock)

    profile_check = commands.add_parser(
        "validate-profile",
        help="validate a frozen baseline success/resource profile",
    )
    profile_check.add_argument("--profile", type=Path, required=True)
    profile_check.add_argument("--registry", type=Path, default=SOURCE_REGISTRY_PATH)
    profile_check.set_defaults(handler=_validate_profile)

    lock_check = commands.add_parser(
        "validate-lock", help="reproduce a lock and recheck the live ledger"
    )
    lock_check.add_argument("--lock", type=Path, required=True)
    lock_check.add_argument("--baseline-profile", type=Path, required=True)
    lock_check.add_argument("--registry", type=Path, default=SOURCE_REGISTRY_PATH)
    lock_check.add_argument("--ledger", type=Path, required=True)
    lock_check.add_argument("--realized-allocation", type=Path)
    lock_check.set_defaults(handler=_validate_lock)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
