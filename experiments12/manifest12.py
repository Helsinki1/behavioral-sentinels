"""Immutable run manifests and reproducibility receipts for Experiment 12."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import (
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_json,
)
from experiments12.models12 import CATALOG_PATH
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    passive_monitor_manifest_binding,
    passive_monitor_spec_from_manifest,
)
from experiments12.spec12 import (
    HARD_PROVIDER_USD,
    OPERATIONAL_PROVIDER_USD,
    Stage,
)


MANIFEST_VERSION = 2
_RUN_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    name: str
    path: str
    sha256: str
    upstream_commit: str | None = None
    license_id: str | None = None
    permission_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.path:
            raise ValueError("artifact receipt needs name/path")
        for name in ("sha256", "permission_receipt_sha256"):
            value = getattr(self, name)
            if value is not None and (
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA256")

    @classmethod
    def from_file(
        cls,
        name: str,
        path: str | Path,
        *,
        workspace: str | Path,
        upstream_commit: str | None = None,
        license_id: str | None = None,
        permission_receipt: str | Path | None = None,
    ) -> "ArtifactReceipt":
        workspace_path = Path(workspace).resolve()
        file_path = Path(path).resolve()
        try:
            relative = file_path.relative_to(workspace_path)
            display = str(relative)
        except ValueError:
            # External dependencies are recorded without exposing a home path.
            display = f"external:{file_path.name}"
        return cls(
            name=name,
            path=display,
            sha256=sha256_file(file_path),
            upstream_commit=upstream_commit,
            license_id=license_id,
            permission_receipt_sha256=(
                None if permission_receipt is None else sha256_file(permission_receipt)
            ),
        )


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path
    manifest: Path
    pairs: Path
    ledger: Path
    events: Path
    trajectories: Path
    shadow: Path
    results: Path

    @classmethod
    def for_run(cls, base: str | Path, run_id: str) -> "RunLayout":
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id contains unsafe characters")
        root = Path(base) / run_id
        global_ledger = Path(base) / "_global_budget.sqlite3"
        return cls(
            root=root,
            manifest=root / "manifest.json",
            pairs=root / "pairs.jsonl",
            # All runs/stages share one authoritative provider ledger; otherwise
            # each $30 Fireworks run could independently spend the full cap.
            ledger=global_ledger,
            events=root / "events",
            trajectories=root / "trajectories",
            shadow=root / "shadow",
            results=root / "results",
        )

    def create(self) -> None:
        for path in (self.root, self.events, self.trajectories, self.shadow, self.results):
            path.mkdir(parents=True, exist_ok=True)


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def code_tree_hash(package_root: str | Path) -> str:
    package = Path(package_root)
    records = []
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".sqlite3"}:
            continue
        if any(part in {"artifacts", "external", "generated"} for part in path.parts):
            continue
        records.append({"path": str(path.relative_to(package)), "sha256": sha256_file(path)})
    return sha256_json(records)


def build_manifest(
    *,
    run_id: str,
    stage: Stage,
    repository_root: str | Path,
    pair_manifest_sha256: str,
    models: Sequence[str],
    arms: Sequence[str],
    operators: Sequence[str],
    randomization_seed: int,
    benchmark_receipts: Sequence[ArtifactReceipt],
    extra_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id contains unsafe characters")
    if len(pair_manifest_sha256) != 64:
        raise ValueError("pair manifest hash must be SHA256")
    if randomization_seed < 0 or not models or not arms or not operators:
        raise ValueError("seed and nonempty model/arm/operator lists are required")
    root = Path(repository_root).resolve()
    package = root / "experiments12"
    dirty_text = _git(root, "status", "--porcelain", "--untracked-files=no")
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "experiment": "12",
        "run_id": run_id,
        "stage": stage.value,
        "created_at": _now(),
        "repository": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "tracked_dirty": bool(dirty_text and dirty_text != "unknown"),
            "code_tree_sha256": code_tree_hash(package),
        },
        "model_catalog_sha256": sha256_file(CATALOG_PATH),
        "pair_manifest_sha256": pair_manifest_sha256,
        "models": list(models),
        "arms": list(arms),
        "operators": list(operators),
        "randomization_seed": randomization_seed,
        "budget_usd": {
            "hard": dict(HARD_PROVIDER_USD),
            "operational": dict(OPERATIONAL_PROVIDER_USD),
        },
        "passive_monitor_spec": passive_monitor_manifest_binding(),
        "benchmark_receipts": [],
        "extra_config": dict(extra_config or {}),
        "secret_values_recorded": False,
    }
    manifest["benchmark_receipts"] = [
        {
            "name": r.name,
            "path": r.path,
            "sha256": r.sha256,
            "upstream_commit": r.upstream_commit,
            "license_id": r.license_id,
            "permission_receipt_sha256": r.permission_receipt_sha256,
        }
        for r in benchmark_receipts
    ]
    return manifest


def write_manifest_once(path: str | Path, manifest: Mapping[str, Any]) -> str:
    """Create an immutable manifest; idempotent only for identical content."""

    destination = Path(path)
    if destination.exists():
        existing = read_json(destination)
        if sha256_json(existing) != sha256_json(dict(manifest)):
            raise FileExistsError("run manifest already exists with different content")
        return sha256_file(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(destination, dict(manifest))


def validate_manifest_files(
    manifest: Mapping[str, Any],
    *,
    repository_root: str | Path,
    pair_manifest_path: str | Path,
) -> tuple[str, ...]:
    errors: list[str] = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        errors.append("unsupported manifest version")
    if sha256_file(pair_manifest_path) != manifest.get("pair_manifest_sha256"):
        errors.append("pair manifest hash mismatch")
    root = Path(repository_root)
    if code_tree_hash(root / "experiments12") != manifest.get("repository", {}).get(
        "code_tree_sha256"
    ):
        errors.append("Experiment 12 code/config tree changed")
    if sha256_file(CATALOG_PATH) != manifest.get("model_catalog_sha256"):
        errors.append("model price catalog changed")
    try:
        passive_monitor_spec_from_manifest(manifest)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if manifest.get("passive_monitor_spec", {}).get("sha256") != PASSIVE_MONITOR_SPEC_SHA256:
            errors.append("passive monitor spec hash mismatch")
    return tuple(errors)
