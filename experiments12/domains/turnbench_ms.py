"""Permission-gated boundary for the unlicensed TurnBench-MS checkout.

No TurnBench prompts, data, verifier code, or game semantics are copied here.
The adapter only validates authorization and provenance, then returns a
subprocess boundary for a separately supplied loader.  The paper's process
extractor and turn-level annotations are not present in the pinned repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Mapping

from .base import (
    ArtifactIntegrityError,
    DomainTask,
    DomainUnavailableError,
    DomainValidationError,
    ExternalLoaderBoundary,
    InputArtifact,
    PermissionGateError,
    read_hashed_file,
    validate_sha256,
)


DOMAIN = "turnbench_ms_classic"
REPOSITORY = "https://github.com/grantzyr/TurnBench-MS"
PINNED_COMMIT = "b3a9daa914e66f62048b62cff06bcaf4151aadb5"
ROOT_ENVIRONMENT_VARIABLE = "TURNBENCH_MS_ROOT"
REQUIRED_PERMISSION_SCOPES = frozenset({"research_use", "local_execution"})

# Digests are for the exact files at PINNED_COMMIT.  They let a source archive
# without .git metadata be checked without redistributing any upstream bytes.
PINNED_PATH_SHA256: Mapping[str, str] = {
    "data/configs/game_setups_45_model_classic.json": (
        "5d368691fe850b9e60f07a4490ced219b5893bda7216c231694bc06b9b9537ae"
    ),
    "data/configs/game_setups_270.json": (
        "24497896af142fffbfa1026c535f15a6cc1f079bf59103384dc072d55edc0de0"
    ),
    "data/configs/verifiers_v1.json": (
        "596ef4499af8ece4500291000e74e6c8af2c756281d4aca4be2563b7dced04c2"
    ),
}

REQUIRED_RUNTIME_PATHS = (
    "verifier/criteria.py",
    "models/game.py",
    "llm/response_parser.py",
    "prompt/templates/system_prompts.py",
    "prompt/templates/proposal_prompts.py",
    "prompt/templates/question_prompts.py",
    "prompt/templates/deduce_prompts.py",
)

_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class TurnBenchReadiness:
    """Auditable result of all gates that do not execute upstream code."""

    root: str
    permission_receipt_sha256: str
    checkout_commit: str | None
    verified_paths: tuple[InputArtifact, ...]
    upstream_license_file: str | None
    official_process_labels_available: bool
    official_process_extractor_available: bool
    notes: tuple[str, ...]

    @property
    def ready_for_external_loader(self) -> bool:
        return bool(self.verified_paths)


def known_release_limitations() -> tuple[str, ...]:
    """Facts that should be copied into every TurnBench run report."""

    return (
        "The pinned TurnBench-MS repository publishes no license file; this adapter "
        "requires a separate permission receipt even if the checkout is present.",
        "The paper's process extractor and process-label annotations are not released "
        "in the pinned repository; this adapter exposes no official turn-level labels.",
    )


def _parse_receipt(path: str | Path) -> tuple[Path, str]:
    try:
        receipt_path, receipt_bytes, receipt_sha256 = read_hashed_file(path)
    except DomainUnavailableError as exc:
        raise PermissionGateError(f"permission receipt is unavailable: {path}") from exc
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionGateError(
            f"permission receipt is not valid JSON: {receipt_path}"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise PermissionGateError("permission receipt must be a JSON object")

    required = {
        "schema_version",
        "repository",
        "commit",
        "permission_granted",
        "scope",
        "granted_by",
        "granted_at",
        "evidence_sha256",
    }
    missing = required - set(receipt)
    if missing:
        raise PermissionGateError(f"permission receipt is missing {sorted(missing)}")
    if (
        isinstance(receipt["schema_version"], bool)
        or not isinstance(receipt["schema_version"], int)
        or receipt["schema_version"] != 1
    ):
        raise PermissionGateError("unsupported permission receipt schema_version")
    if (
        not isinstance(receipt["repository"], str)
        or receipt["repository"].rstrip("/") != REPOSITORY
    ):
        raise PermissionGateError("permission receipt names a different repository")
    if receipt["commit"] != PINNED_COMMIT:
        raise PermissionGateError("permission receipt does not cover the pinned commit")
    if receipt["permission_granted"] is not True:
        raise PermissionGateError("permission receipt does not affirm permission")

    scope = receipt["scope"]
    if not isinstance(scope, list) or not all(
        isinstance(item, str) and item.strip() for item in scope
    ):
        raise PermissionGateError("permission receipt scope must be a string array")
    normalized_scope = {item.strip().lower() for item in scope}
    missing_scope = REQUIRED_PERMISSION_SCOPES - normalized_scope
    if missing_scope:
        raise PermissionGateError(
            f"permission receipt does not cover required scopes: {sorted(missing_scope)}"
        )
    for field in ("granted_by", "granted_at"):
        if not isinstance(receipt[field], str) or not receipt[field].strip():
            raise PermissionGateError(f"permission receipt {field} must be non-empty")
    try:
        validate_sha256("evidence_sha256", receipt["evidence_sha256"])
    except DomainValidationError as exc:
        raise PermissionGateError(str(exc)) from exc
    return receipt_path, receipt_sha256


def _git_directory(root: Path) -> Path | None:
    marker = root / ".git"
    if not marker.exists():
        return None
    if marker.is_dir():
        return marker
    try:
        content = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ArtifactIntegrityError(f"could not read git metadata: {marker}") from exc
    prefix = "gitdir:"
    if not content.lower().startswith(prefix):
        raise ArtifactIntegrityError(f"invalid gitdir marker: {marker}")
    git_dir = Path(content[len(prefix) :].strip())
    if not git_dir.is_absolute():
        git_dir = marker.parent / git_dir
    try:
        return git_dir.resolve(strict=True)
    except OSError as exc:
        raise ArtifactIntegrityError(f"git directory is unavailable: {git_dir}") from exc


def _read_git_head(root: Path) -> str | None:
    git_dir = _git_directory(root)
    if git_dir is None:
        return None
    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ArtifactIntegrityError(f"could not read checkout HEAD: {head_path}") from exc
    if _GIT_OBJECT_RE.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        raise ArtifactIntegrityError("checkout HEAD is neither a commit nor a ref")
    ref = head[5:].strip()
    ref_path = Path(ref)
    if (
        not ref
        or ref_path.is_absolute()
        or ".." in ref_path.parts
        or not ref_path.parts
        or ref_path.parts[0] != "refs"
    ):
        raise ArtifactIntegrityError("checkout HEAD contains an invalid ref")
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        value = loose_ref.read_text(encoding="utf-8").strip()
        if not _GIT_OBJECT_RE.fullmatch(value):
            raise ArtifactIntegrityError(f"invalid object ID in {loose_ref}")
        return value
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            try:
                value, candidate_ref = line.split(" ", 1)
            except ValueError:
                continue
            if candidate_ref == ref and _GIT_OBJECT_RE.fullmatch(value):
                return value
    raise ArtifactIntegrityError(f"could not resolve checkout HEAD ref: {ref}")


class TurnBenchMSAdapter:
    """Validate authorization/provenance and expose an out-of-process boundary."""

    domain = DOMAIN

    def __init__(
        self,
        permission_receipt_path: str | Path,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        receipt_path, receipt_sha256 = _parse_receipt(permission_receipt_path)
        env = os.environ if environment is None else environment
        root_value = env.get(ROOT_ENVIRONMENT_VARIABLE)
        if not isinstance(root_value, str) or not root_value.strip():
            raise PermissionGateError(
                f"{ROOT_ENVIRONMENT_VARIABLE} must name the authorized external checkout"
            )
        try:
            root = Path(root_value).expanduser().resolve(strict=True)
        except OSError as exc:
            raise DomainUnavailableError(
                f"TurnBench external checkout is unavailable: {root_value}"
            ) from exc
        if not root.is_dir():
            raise DomainUnavailableError(f"TurnBench root is not a directory: {root}")

        checkout_commit = _read_git_head(root)
        if checkout_commit is not None and checkout_commit != PINNED_COMMIT:
            raise ArtifactIntegrityError(
                f"TurnBench checkout must be {PINNED_COMMIT}, got {checkout_commit}"
            )

        verified: list[InputArtifact] = []
        verified_relative_paths: set[str] = set()
        for relative_path, expected_digest in PINNED_PATH_SHA256.items():
            resolved, _data, digest = read_hashed_file(
                root / relative_path, expected_sha256=expected_digest
            )
            verified.append(InputArtifact(f"turnbench:{relative_path}", str(resolved), digest))
            verified_relative_paths.add(relative_path)
        for relative_path in REQUIRED_RUNTIME_PATHS:
            if relative_path in verified_relative_paths:
                continue
            resolved, _data, digest = read_hashed_file(root / relative_path)
            verified.append(
                InputArtifact(f"turnbench:{relative_path}", str(resolved), digest)
            )

        license_file = next(
            (
                str(candidate.resolve())
                for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
                if (candidate := root / name).is_file()
            ),
            None,
        )
        notes = list(known_release_limitations())
        if license_file is not None:
            notes[0] = (
                "A license-like file exists in this local checkout, but the pinned release "
                "did not publish one; the explicit permission receipt remains authoritative."
            )
        receipt_artifact = InputArtifact(
            "turnbench_permission_receipt", str(receipt_path), receipt_sha256
        )
        self._input_artifacts = (receipt_artifact, *verified)
        self._root = root
        self._readiness = TurnBenchReadiness(
            root=str(root),
            permission_receipt_sha256=receipt_sha256,
            checkout_commit=checkout_commit,
            verified_paths=tuple(verified),
            upstream_license_file=license_file,
            official_process_labels_available=False,
            official_process_extractor_available=False,
            notes=tuple(notes),
        )

    @property
    def input_artifacts(self) -> tuple[InputArtifact, ...]:
        return self._input_artifacts

    @property
    def readiness(self) -> TurnBenchReadiness:
        return self._readiness

    def loader_boundary(self) -> ExternalLoaderBoundary:
        """Return a process boundary; never import unlicensed code in-process."""

        # The external loader will read live files, so close the time-of-check /
        # time-of-use gap before handing them to another process.
        for artifact in self._input_artifacts:
            read_hashed_file(artifact.path, expected_sha256=artifact.sha256)
        checkout_commit = _read_git_head(self._root)
        if checkout_commit is not None and checkout_commit != PINNED_COMMIT:
            raise ArtifactIntegrityError(
                f"TurnBench checkout changed revisions after validation: {checkout_commit}"
            )
        return ExternalLoaderBoundary(
            external_root=str(self._root),
            root_environment_variable=ROOT_ENVIRONMENT_VARIABLE,
            pinned_commit=PINNED_COMMIT,
            verified_inputs=self._readiness.verified_paths,
        )

    def load_tasks(self) -> tuple[DomainTask, ...]:
        """Fail explicitly instead of silently recreating TurnBench semantics."""

        raise DomainUnavailableError(
            "TurnBench is interactive and must be driven through loader_boundary() by "
            "authorized upstream code. No official process extractor or process-label "
            "artifact is available, so this adapter cannot emit static DomainTask records."
        )
