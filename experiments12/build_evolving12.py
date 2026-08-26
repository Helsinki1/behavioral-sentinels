"""Reproducible, resumable Evolving Intent GSM8K reproduction builder.

The upstream repository does not publish its generated ``gsm8k_final.json`` and
its generation scripts call an SDK directly.  This module therefore does not
copy or approximate those semantics.  An explicit bridge kept inside the
pinned MIT checkout must expose upstream prompt/validation state as a small JSON
state machine:

* ``capabilities`` proves that the bridge emits requests rather than making
  network calls itself;
* ``advance_stage`` executes one of the three upstream stages until it either
  needs one generator/judge call or returns a validated stage artifact;
* ``render_pair`` invokes the upstream rule-based simulator for t=1 and for
  t=7, g=2, p=2.

Only this process calls :class:`experiments12.core.transport.Transport`, so all
provider attempts remain budgeted and audited.  Provider responses and opaque
upstream state are atomically checkpointed per task.  A pending call with no
saved response fails closed on resume because its billing/outcome is unknown.

No upstream source, prompt, or generated datum is vendored into this package.
The final artifact is shared across every target model and experimental arm.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Protocol, Sequence

from experiments12.core.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from experiments12.core.budget import BudgetLedger
from experiments12.core.transport import (
    CompletionResult,
    JsonSchemaOutput,
    Transport,
    TransportError,
)
from experiments12.cli12 import DEFAULT_ARTIFACTS, REPOSITORY_ROOT, _environment
from experiments12.domains.base import InputArtifact
from experiments12.domains.evolving_intent import (
    DOMAIN,
    EvolvingIntentAdapter,
    PINNED_COMMIT,
)
from experiments12.models12 import CATALOG
from experiments12.spec12 import OPERATIONAL_PROVIDER_USD, STAGE_PROVIDER_USD, Stage


ROOT_ENVIRONMENT_VARIABLE = "EVOLVING_INTENT_ROOT"
BRIDGE_PROTOCOL = "behavioral-sentinels.evolving-intent-bridge.v1"
BUILD_SCHEMA_VERSION = 1
SEED = 42
BRIDGE_DEPENDENCY_LOCK = Path(__file__).resolve().with_name(
    "evolving_intent_math_verify.lock"
)
DEFAULT_BRIDGE_PYTHON_RELATIVE = Path(".venv/bin/python")
BRIDGE_PYTHON_MAJOR_MINOR = (3, 12)
BRIDGE_COMPATIBILITY_PATCHES = (
    {
        "id": "gsm8k_similarity_placeholder_names",
        "upstream_prompt": "similarity_check_gsm8k.txt",
        "repair": "goal_a/goal_b -> function_a/function_b",
    },
)
PROVIDER_COMPATIBILITY = {
    "openai_min_output_tokens": 16,
    "applies_when_upstream_limit_is_lower": True,
}
BRIDGE_INVOCATION_FLAGS = ("-s", "-P")
BRIDGE_RUNTIME_DEPENDENCIES: Mapping[str, str] = {
    "antlr4-python3-runtime": "4.13.2",
    "latex2sympy2-extended": "1.11.0",
    "math-verify": "0.9.0",
    "mpmath": "1.3.0",
    "sympy": "1.14.0",
}
STAGES = (
    "intent_extraction",
    "counterfactual_generation",
    "predecessor_generation",
)
MAX_BRIDGE_STEPS_PER_STAGE = 256
MAX_BRIDGE_RESPONSE_BYTES = 32 * 1024 * 1024

OFFICIAL_IDS_PATH = "intent_construction/eval_indices/gsm8k_eval_ids.json"
REQUIRED_UPSTREAM_PATHS = (
    "LICENSE",
    OFFICIAL_IDS_PATH,
    "intent_construction/intent_extraction",
    "intent_construction/retrospective_expansion/counterfactual",
    "intent_construction/retrospective_expansion/predecessor",
    "situated_simulation/user_simulation.py",
)
STAGE_SOURCE_ROOTS = (
    "intent_construction/intent_extraction",
    "intent_construction/retrospective_expansion/counterfactual",
    "intent_construction/retrospective_expansion/predecessor",
    "situated_simulation",
)
HASHED_SOURCE_SUFFIXES = frozenset({".py", ".json", ".txt", ".md", ".yaml", ".yml"})

# This descriptor is deliberately data, not prose only: an external patch must
# attest its exact hash before the builder authorizes any provider request.
BRIDGE_CONTRACT: Mapping[str, Any] = {
    "protocol": BRIDGE_PROTOCOL,
    "capabilities": {
        "required": [
            "protocol",
            "contract_sha256",
            "upstream_commit",
            "transport_mode",
            "stages",
            "renderer",
        ]
    },
    "advance_stage": {
        "request": [
            "stage",
            "source_task",
            "prior_artifacts",
            "state",
            "model_result",
        ],
        "responses": {
            "needs_model_call": ["state", "call"],
            "complete": ["artifact"],
        },
        "call": [
            "call_key",
            "role",
            "messages",
            "temperature",
            "max_output_tokens",
            "output_schema",
            "prompt_files",
        ],
    },
    "render_pair": {
        "request": ["source_task", "stage_artifacts", "conditions", "seed"],
        "response": ["status=rendered", "simulator.kind=rule_based", "records=t1+t7"],
    },
}
BRIDGE_CONTRACT_SHA256 = sha256_json(BRIDGE_CONTRACT)

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_CALL_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")


class ReproductionError(RuntimeError):
    """Base class for a deterministic build failure."""


class ReadinessBlocked(ReproductionError):
    """The external checkout cannot yet support a faithful reproduction."""


class ResumeBlocked(ReproductionError):
    """A prior run stopped at a point that cannot be safely replayed."""


class BridgeProtocolError(ReproductionError):
    """The checkout-local bridge violated the declared JSON protocol."""


class BridgeRuntimeError(ReadinessBlocked):
    """The selected checkout-local Python runtime is unavailable or changed."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ReadinessIssue:
    code: str
    detail: str
    blocking: bool

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "blocking": self.blocking}


@dataclass(frozen=True, slots=True)
class BridgeRuntimeAttestation:
    """Frozen identity of the isolated interpreter used for every bridge call."""

    python_path: str
    resolved_executable_path: str
    executable_sha256: str
    python_version: str
    python_version_info: tuple[int, int, int]
    implementation: str
    venv_path: str
    base_prefix: str
    pyvenv_config_path: str
    pyvenv_config_sha256: str
    dependency_lock_path: str
    dependency_lock_sha256: str

    def _core_dict(self) -> dict[str, Any]:
        return {
            "python_path": self.python_path,
            "resolved_executable_path": self.resolved_executable_path,
            "executable_sha256": self.executable_sha256,
            "python_version": self.python_version,
            "python_version_info": list(self.python_version_info),
            "implementation": self.implementation,
            "venv_path": self.venv_path,
            "base_prefix": self.base_prefix,
            "pyvenv_config_path": self.pyvenv_config_path,
            "pyvenv_config_sha256": self.pyvenv_config_sha256,
            "dependency_lock_path": self.dependency_lock_path,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "locked_dependencies": dict(BRIDGE_RUNTIME_DEPENDENCIES),
        }

    @property
    def attestation_sha256(self) -> str:
        return sha256_json(self._core_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._core_dict(),
            "attestation_sha256": self.attestation_sha256,
        }

    @property
    def input_artifacts(self) -> tuple[InputArtifact, ...]:
        return (
            InputArtifact(
                "bridge_python_executable", self.python_path, self.executable_sha256
            ),
            InputArtifact(
                "bridge_python_venv_config",
                self.pyvenv_config_path,
                self.pyvenv_config_sha256,
            ),
        )


@dataclass(frozen=True, slots=True)
class ReproductionReadiness:
    root: str | None
    checkout_commit: str | None
    bridge_path: str | None
    bridge_runtime: BridgeRuntimeAttestation | None
    input_artifacts: tuple[InputArtifact, ...]
    prompt_artifacts: tuple[InputArtifact, ...]
    issues: tuple[ReadinessIssue, ...]

    @property
    def ready(self) -> bool:
        return not any(issue.blocking for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BUILD_SCHEMA_VERSION,
            "benchmark": DOMAIN,
            "upstream_commit": PINNED_COMMIT,
            "root": self.root,
            "checkout_commit": self.checkout_commit,
            "bridge_path": self.bridge_path,
            "bridge_runtime": (
                None if self.bridge_runtime is None else self.bridge_runtime.as_dict()
            ),
            "ready": self.ready,
            "input_artifacts": [
                {"role": item.role, "path": item.path, "sha256": item.sha256}
                for item in self.input_artifacts
            ],
            "prompt_artifacts": [
                {"role": item.role, "path": item.path, "sha256": item.sha256}
                for item in self.prompt_artifacts
            ],
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class OfficialTaskID:
    source_id: int
    task_id: str


@dataclass(frozen=True, slots=True)
class SourceTask:
    source_id: int
    task_id: str
    question: str
    answer: str

    def private_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "task_id": self.task_id,
            "question": self.question,
            "answer": self.answer,
        }


@dataclass(frozen=True, slots=True)
class BuildSettings:
    generator_model: str
    judge_model: str
    generator_reasoning_effort: str | None = None
    judge_reasoning_effort: str | None = None
    seed: int = SEED

    def __post_init__(self) -> None:
        for name in ("generator_model", "judge_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("generator_reasoning_effort", "judge_reasoning_effort"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty or None")
        if self.seed != SEED:
            raise ValueError(f"Evolving Intent reproduction seed is fixed at {SEED}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "generator_model": self.generator_model,
            "judge_model": self.judge_model,
            "generator_reasoning_effort": self.generator_reasoning_effort,
            "judge_reasoning_effort": self.judge_reasoning_effort,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class BuildResult:
    dataset_path: str
    dataset_sha256: str
    receipt_path: str
    receipt_sha256: str
    num_source_tasks: int
    num_condition_records: int


class GenerationTransport(Protocol):
    async def complete(
        self,
        model_name: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult: ...


class UpstreamBridge(Protocol):
    @property
    def artifact(self) -> InputArtifact: ...

    @property
    def runtime(self) -> BridgeRuntimeAttestation: ...

    def exchange(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def parse_ids(value: str) -> tuple[int, ...]:
    """Parse zero-based GSM8K source row IDs, preserving declared order."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("--ids must be a comma-separated list")
    result: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item or not item.isdigit():
            raise ValueError(f"invalid source ID: {raw!r}")
        source_id = int(item)
        if source_id in result:
            raise ValueError(f"duplicate source ID: {source_id}")
        result.append(source_id)
    return tuple(result)


def _resolve_git_dir(root: Path) -> Path | None:
    marker = root / ".git"
    if not marker.exists():
        return None
    if marker.is_dir():
        return marker.resolve()
    try:
        marker_text = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not marker_text.lower().startswith("gitdir:"):
        return None
    candidate = Path(marker_text.split(":", 1)[1].strip())
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError:
        return None


def _checkout_head(root: Path) -> str | None:
    git_dir = _resolve_git_dir(root)
    if git_dir is None:
        return None
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if _HEX40_RE.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        return None
    ref = head[5:].strip()
    ref_path = Path(ref)
    if ref_path.is_absolute() or ".." in ref_path.parts or not ref.startswith("refs/"):
        return None
    loose = git_dir / ref
    if loose.is_file():
        try:
            value = loose.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value if _HEX40_RE.fullmatch(value) else None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        try:
            lines = packed.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            if not line or line.startswith(("#", "^")):
                continue
            fields = line.split(" ", 1)
            if len(fields) == 2 and fields[1] == ref and _HEX40_RE.fullmatch(fields[0]):
                return fields[0]
    return None


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return candidate.resolve(strict=True).is_relative_to(root.resolve(strict=True))
    except OSError:
        return False


_RUNTIME_PROBE = (
    "import json,platform,sys;"
    "print(json.dumps({"
    "'base_prefix':sys.base_prefix,"
    "'executable':sys.executable,"
    "'implementation':platform.python_implementation().lower(),"
    "'prefix':sys.prefix,"
    "'version':platform.python_version(),"
    "'version_info':list(sys.version_info[:3])"
    "},sort_keys=True))"
)


def _absolute_without_resolving(path: str | Path) -> Path:
    """Return a normalized absolute path while retaining its venv symlink entry."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _attest_bridge_runtime(
    root: str | Path,
    python_path: str | Path,
) -> BridgeRuntimeAttestation:
    """Validate and fingerprint the checkout-local CPython virtual environment."""

    try:
        resolved_root = Path(root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise BridgeRuntimeError(
            "bridge_python_root_unavailable", "the pinned checkout root is unavailable"
        ) from exc
    venv_path = resolved_root / ".venv"
    if venv_path.is_symlink():
        raise BridgeRuntimeError(
            "bridge_python_venv_symlink",
            "the checkout-local .venv directory must not be a symlink",
        )
    try:
        resolved_venv = venv_path.resolve(strict=True)
    except OSError as exc:
        raise BridgeRuntimeError(
            "bridge_python_venv_missing",
            f"create the locked checkout-local virtual environment at {venv_path}",
        ) from exc
    if not resolved_venv.is_dir() or not resolved_venv.is_relative_to(resolved_root):
        raise BridgeRuntimeError(
            "bridge_python_venv_outside_checkout",
            "the bridge virtual environment must be a directory inside the pinned checkout",
        )

    invocation_path = _absolute_without_resolving(python_path)
    allowed_bin = venv_path / "bin"
    if not invocation_path.is_relative_to(allowed_bin):
        raise BridgeRuntimeError(
            "bridge_python_outside_checkout_venv",
            "the selected bridge interpreter must be an entry under EVOLVING_INTENT_ROOT/.venv/bin",
        )
    if not invocation_path.exists() or not invocation_path.is_file():
        raise BridgeRuntimeError(
            "bridge_python_unavailable", f"bridge interpreter is unavailable: {invocation_path}"
        )
    if not os.access(invocation_path, os.X_OK):
        raise BridgeRuntimeError(
            "bridge_python_not_executable",
            f"bridge interpreter is not executable: {invocation_path}",
        )
    try:
        resolved_executable = invocation_path.resolve(strict=True)
    except OSError as exc:
        raise BridgeRuntimeError(
            "bridge_python_unavailable", f"bridge interpreter is unavailable: {invocation_path}"
        ) from exc

    pyvenv_config = venv_path / "pyvenv.cfg"
    if not pyvenv_config.is_file():
        raise BridgeRuntimeError(
            "bridge_python_venv_config_missing",
            f"virtual-environment configuration is unavailable: {pyvenv_config}",
        )
    if not BRIDGE_DEPENDENCY_LOCK.is_file():
        raise BridgeRuntimeError(
            "bridge_dependency_lock_missing",
            f"bridge dependency lock is unavailable: {BRIDGE_DEPENDENCY_LOCK}",
        )

    environment = {
        "PATH": os.defpath,
        "PYTHONHASHSEED": str(SEED),
        "LC_ALL": "C.UTF-8",
    }
    try:
        completed = subprocess.run(
            (str(invocation_path), "-I", "-c", _RUNTIME_PROBE),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=resolved_root,
            env=environment,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BridgeRuntimeError(
            "bridge_python_probe_failed", "the selected bridge interpreter could not be probed"
        ) from exc
    if completed.returncode != 0 or len(completed.stdout) > 16 * 1024:
        raise BridgeRuntimeError(
            "bridge_python_probe_failed", "the selected bridge interpreter failed its probe"
        )
    try:
        probe = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeRuntimeError(
            "bridge_python_probe_invalid", "the bridge interpreter returned invalid probe data"
        ) from exc
    if not isinstance(probe, Mapping):
        raise BridgeRuntimeError(
            "bridge_python_probe_invalid", "the bridge interpreter returned invalid probe data"
        )
    required_strings = (
        "base_prefix",
        "executable",
        "implementation",
        "prefix",
        "version",
    )
    if any(not isinstance(probe.get(name), str) or not probe[name] for name in required_strings):
        raise BridgeRuntimeError(
            "bridge_python_probe_invalid", "the bridge interpreter probe is incomplete"
        )
    version_info = probe.get("version_info")
    if (
        not isinstance(version_info, list)
        or len(version_info) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in version_info)
    ):
        raise BridgeRuntimeError(
            "bridge_python_probe_invalid", "the bridge interpreter version tuple is invalid"
        )
    if tuple(version_info[:2]) != BRIDGE_PYTHON_MAJOR_MINOR:
        raise BridgeRuntimeError(
            "bridge_python_version_mismatch",
            "the dependency lock is resolved for CPython "
            f"{BRIDGE_PYTHON_MAJOR_MINOR[0]}.{BRIDGE_PYTHON_MAJOR_MINOR[1]}, "
            f"not {version_info[0]}.{version_info[1]}",
        )
    if probe["implementation"] != "cpython":
        raise BridgeRuntimeError(
            "bridge_python_implementation_mismatch",
            "the Evolving Intent dependency lock requires CPython",
        )
    try:
        reported_prefix = Path(probe["prefix"]).resolve(strict=True)
        reported_executable = Path(probe["executable"]).resolve(strict=True)
        base_prefix = str(Path(probe["base_prefix"]).resolve(strict=True))
    except OSError as exc:
        raise BridgeRuntimeError(
            "bridge_python_probe_invalid", "the bridge interpreter reported unavailable paths"
        ) from exc
    if reported_prefix != resolved_venv or probe["prefix"] == probe["base_prefix"]:
        raise BridgeRuntimeError(
            "bridge_python_runtime_mismatch",
            "the selected interpreter is not running from EVOLVING_INTENT_ROOT/.venv",
        )
    if reported_executable != resolved_executable:
        raise BridgeRuntimeError(
            "bridge_python_runtime_mismatch",
            "the selected interpreter resolved to a different runtime executable",
        )

    return BridgeRuntimeAttestation(
        python_path=str(invocation_path),
        resolved_executable_path=str(resolved_executable),
        executable_sha256=sha256_file(invocation_path),
        python_version=probe["version"],
        python_version_info=tuple(version_info),
        implementation=probe["implementation"],
        venv_path=str(resolved_venv),
        base_prefix=base_prefix,
        pyvenv_config_path=str(pyvenv_config),
        pyvenv_config_sha256=sha256_file(pyvenv_config),
        dependency_lock_path=str(BRIDGE_DEPENDENCY_LOCK),
        dependency_lock_sha256=sha256_file(BRIDGE_DEPENDENCY_LOCK),
    )


def _source_artifacts(root: Path) -> tuple[InputArtifact, ...]:
    artifacts: list[InputArtifact] = []
    seen: set[Path] = set()
    for relative_root in STAGE_SOURCE_ROOTS:
        source_root = root / relative_root
        if not source_root.exists():
            continue
        candidates = (source_root,) if source_root.is_file() else source_root.rglob("*")
        for path in candidates:
            if (
                not path.is_file()
                or not _inside(root, path)
                or path.suffix.lower() not in HASHED_SOURCE_SUFFIXES
            ):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            role = "upstream_prompt" if "prompt" in path.name.lower() else "upstream_stage_source"
            artifacts.append(InputArtifact(role, str(resolved), sha256_file(resolved)))
    return tuple(sorted(artifacts, key=lambda item: item.path))


def audit_reproduction(
    *,
    gsm8k_test_path: str | Path | None = None,
    bridge_path: str | Path | None = None,
    bridge_python_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> ReproductionReadiness:
    """Audit provenance and explicitly enumerate all remaining blockers."""

    env = os.environ if environment is None else environment
    issues: list[ReadinessIssue] = [
        ReadinessIssue(
            "upstream_generated_dataset_unreleased",
            "The official generated gsm8k_final.json is not published; any successful "
            "build is a frozen reproduction, not a byte-identical official artifact.",
            False,
        ),
        ReadinessIssue(
            "generator_snapshot_must_be_reported",
            "The released construction is stochastic and model-version dependent; the "
            "chosen generator/judge IDs and settings will be recorded as reproduction "
            "metadata rather than claimed as an unreleased official snapshot.",
            False,
        ),
        ReadinessIssue(
            "provider_sampling_not_seeded",
            "Seed 42 fixes source selection, upstream bridge randomness, and simulator "
            "scheduling. Experiment 12 Transport exposes no provider seed, so stochastic "
            "generation is made reproducible by freezing outputs, not by claiming an "
            "identical rerun.",
            False,
        ),
    ]
    artifacts: list[InputArtifact] = []
    prompt_artifacts: tuple[InputArtifact, ...] = ()
    bridge_runtime: BridgeRuntimeAttestation | None = None
    artifacts.append(
        InputArtifact(
            "experiment12_reproduction_builder",
            str(Path(__file__).resolve()),
            sha256_file(Path(__file__).resolve()),
        )
    )
    root_value = env.get(ROOT_ENVIRONMENT_VARIABLE)
    if not isinstance(root_value, str) or not root_value.strip():
        issues.append(
            ReadinessIssue(
                "external_root_missing",
                f"Set {ROOT_ENVIRONMENT_VARIABLE} to the pinned external MIT checkout.",
                True,
            )
        )
        return ReproductionReadiness(None, None, None, None, (), (), tuple(issues))
    try:
        root = Path(root_value).expanduser().resolve(strict=True)
    except OSError:
        issues.append(ReadinessIssue("external_root_unavailable", str(root_value), True))
        return ReproductionReadiness(None, None, None, None, (), (), tuple(issues))
    if not root.is_dir():
        issues.append(ReadinessIssue("external_root_not_directory", str(root), True))

    if not BRIDGE_DEPENDENCY_LOCK.is_file():
        issues.append(
            ReadinessIssue(
                "bridge_dependency_lock_missing",
                str(BRIDGE_DEPENDENCY_LOCK),
                True,
            )
        )
    else:
        artifacts.append(
            InputArtifact(
                "bridge_dependency_lock",
                str(BRIDGE_DEPENDENCY_LOCK),
                sha256_file(BRIDGE_DEPENDENCY_LOCK),
            )
        )

    if bridge_python_path is None:
        selected_python: str | Path = root / DEFAULT_BRIDGE_PYTHON_RELATIVE
    else:
        selected_python = Path(bridge_python_path).expanduser()
        if not selected_python.is_absolute():
            selected_python = root / selected_python
    try:
        bridge_runtime = _attest_bridge_runtime(root, selected_python)
    except BridgeRuntimeError as exc:
        issues.append(ReadinessIssue(exc.code, exc.detail, True))
    else:
        artifacts.extend(bridge_runtime.input_artifacts)

    commit = _checkout_head(root)
    if commit is None:
        issues.append(
            ReadinessIssue(
                "checkout_commit_unverifiable",
                "The checkout needs readable .git metadata so the pinned commit can be verified.",
                True,
            )
        )
    elif commit != PINNED_COMMIT:
        issues.append(
            ReadinessIssue(
                "checkout_commit_mismatch",
                f"Expected {PINNED_COMMIT}, found {commit}.",
                True,
            )
        )

    for relative_path in REQUIRED_UPSTREAM_PATHS:
        candidate = root / relative_path
        if not candidate.exists():
            issues.append(
                ReadinessIssue("upstream_path_missing", relative_path, True)
            )
        elif candidate.is_file():
            artifacts.append(
                InputArtifact(f"upstream:{relative_path}", str(candidate.resolve()), sha256_file(candidate))
            )

    license_path = root / "LICENSE"
    if license_path.is_file():
        try:
            license_text = license_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            license_text = ""
        if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
            issues.append(
                ReadinessIssue(
                    "mit_license_unverified",
                    "The checkout LICENSE does not contain the standard MIT grant.",
                    True,
                )
            )

    stage_artifacts = _source_artifacts(root)
    prompt_artifacts = tuple(
        artifact for artifact in stage_artifacts if artifact.role == "upstream_prompt"
    )
    artifacts.extend(stage_artifacts)
    if not prompt_artifacts:
        issues.append(
            ReadinessIssue(
                "prompt_files_unidentified",
                "No prompt-named files were found in the three upstream stage trees.",
                True,
            )
        )

    resolved_bridge: Path | None = None
    if bridge_path is None:
        issues.append(
            ReadinessIssue(
                "external_bridge_missing",
                "Supply a checkout-local bridge implementing the documented JSON state machine; "
                "the official repository does not include this transport patch.",
                True,
            )
        )
    else:
        try:
            resolved_bridge = Path(bridge_path).expanduser().resolve(strict=True)
        except OSError:
            issues.append(ReadinessIssue("external_bridge_unavailable", str(bridge_path), True))
        else:
            if not resolved_bridge.is_file() or not _inside(root, resolved_bridge):
                issues.append(
                    ReadinessIssue(
                        "external_bridge_outside_checkout",
                        "The bridge must be a file inside EVOLVING_INTENT_ROOT.",
                        True,
                    )
                )
            else:
                artifacts.append(
                    InputArtifact("upstream_transport_bridge", str(resolved_bridge), sha256_file(resolved_bridge))
                )

    if gsm8k_test_path is not None:
        try:
            source = Path(gsm8k_test_path).expanduser().resolve(strict=True)
        except OSError:
            issues.append(ReadinessIssue("gsm8k_test_unavailable", str(gsm8k_test_path), True))
        else:
            if not source.is_file():
                issues.append(ReadinessIssue("gsm8k_test_not_file", str(source), True))
            else:
                artifacts.append(InputArtifact("gsm8k_test_jsonl", str(source), sha256_file(source)))

    # Deduplicate files reached through both the required list and recursive scan.
    unique = {(item.role, item.path): item for item in artifacts}
    return ReproductionReadiness(
        root=str(root),
        checkout_commit=commit,
        bridge_path=None if resolved_bridge is None else str(resolved_bridge),
        bridge_runtime=bridge_runtime,
        input_artifacts=tuple(sorted(unique.values(), key=lambda item: (item.role, item.path))),
        prompt_artifacts=prompt_artifacts,
        issues=tuple(issues),
    )


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReproductionError(f"invalid JSON file: {path}") from exc


def load_official_id_map(path: str | Path) -> tuple[OfficialTaskID, ...]:
    """Load either released mapping layout without guessing task contents."""

    source = Path(path)
    raw = _json_file(source)
    pairs: list[OfficialTaskID] = []

    def add(source_id: Any, task_id: Any) -> None:
        if isinstance(source_id, str) and source_id.isdigit():
            source_id = int(source_id)
        if isinstance(source_id, bool) or not isinstance(source_id, int) or source_id < 0:
            raise ReproductionError(f"invalid original_id in {source}")
        if isinstance(task_id, bool) or not isinstance(task_id, (str, int)):
            raise ReproductionError(f"invalid task_id in {source}")
        normalized = str(task_id).strip()
        if not normalized:
            raise ReproductionError(f"empty task_id in {source}")
        pairs.append(OfficialTaskID(source_id, normalized))

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                raise ReproductionError("official ID list entries must be objects")
            add(item.get("original_id", item.get("source_id")), item.get("task_id"))
    elif isinstance(raw, Mapping):
        nested = raw.get("samples", raw.get("tasks", raw.get("eval_ids")))
        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, Mapping):
                    raise ReproductionError("official ID entries must be objects")
                add(item.get("original_id", item.get("source_id")), item.get("task_id"))
        else:
            for key, value in raw.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    add(value, key)
                elif isinstance(value, Mapping):
                    add(
                        value.get("original_id", value.get("source_id")),
                        value.get("task_id", key),
                    )
                elif isinstance(value, str) and value.isdigit():
                    add(int(value), key)
                elif isinstance(key, str) and key.isdigit() and isinstance(value, str):
                    add(int(key), value)
                else:
                    raise ReproductionError("unrecognized official ID mapping layout")
    else:
        raise ReproductionError("official ID manifest must be an object or array")

    if not pairs:
        raise ReproductionError("official ID manifest is empty")
    if len({item.source_id for item in pairs}) != len(pairs):
        raise ReproductionError("official ID manifest duplicates source IDs")
    if len({item.task_id for item in pairs}) != len(pairs):
        raise ReproductionError("official ID manifest duplicates task IDs")
    return tuple(pairs)


def _read_jsonl_exact(path: Path) -> tuple[list[Any], str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReproductionError(f"could not read GSM8K JSONL: {path}") from exc
    if data and not data.endswith(b"\n"):
        raise ReproductionError("GSM8K JSONL has a torn final line")
    rows: list[Any] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            raise ReproductionError(f"blank GSM8K JSONL line {line_number}")
        try:
            rows.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReproductionError(f"invalid GSM8K JSONL line {line_number}") from exc
    return rows, sha256_bytes(data)


def select_source_tasks(
    gsm8k_test_path: str | Path,
    official_ids_path: str | Path,
    source_ids: Sequence[int],
) -> tuple[tuple[SourceTask, ...], tuple[InputArtifact, ...]]:
    """Select declared zero-based rows and bind them to released task IDs."""

    if isinstance(source_ids, (str, bytes)) or not source_ids:
        raise ValueError("source_ids must be a non-empty sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in source_ids):
        raise ValueError("source IDs must be non-negative integers")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source IDs must be unique")
    gsm8k_path = Path(gsm8k_test_path).expanduser().resolve(strict=True)
    ids_path = Path(official_ids_path).expanduser().resolve(strict=True)
    rows, dataset_hash = _read_jsonl_exact(gsm8k_path)
    official = {item.source_id: item.task_id for item in load_official_id_map(ids_path)}
    selected: list[SourceTask] = []
    for source_id in source_ids:
        if source_id not in official:
            raise ReproductionError(f"source ID {source_id} is not in the official evaluation set")
        if source_id >= len(rows):
            raise ReproductionError(f"source ID {source_id} is outside supplied GSM8K test JSONL")
        row = rows[source_id]
        if not isinstance(row, Mapping):
            raise ReproductionError(f"GSM8K row {source_id} must be an object")
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ReproductionError(f"GSM8K row {source_id} has no question")
        if not isinstance(answer, str) or not answer.strip():
            raise ReproductionError(f"GSM8K row {source_id} has no answer")
        selected.append(SourceTask(source_id, official[source_id], question, answer))
    return (
        tuple(selected),
        (
            InputArtifact("gsm8k_test_jsonl", str(gsm8k_path), dataset_hash),
            InputArtifact("official_eval_ids", str(ids_path), sha256_file(ids_path)),
        ),
    )


class SubprocessUpstreamBridge:
    """JSON stdin/stdout boundary for code living in the external checkout."""

    def __init__(
        self,
        root: str | Path,
        bridge_path: str | Path,
        python_path: str | Path,
        *,
        expected_runtime: BridgeRuntimeAttestation | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        self.path = Path(bridge_path).expanduser().resolve(strict=True)
        if not self.path.is_file() or not _inside(self.root, self.path):
            raise ReadinessBlocked("bridge must be a file inside EVOLVING_INTENT_ROOT")
        if timeout_seconds <= 0:
            raise ValueError("bridge timeout must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self._artifact = InputArtifact(
            "upstream_transport_bridge", str(self.path), sha256_file(self.path)
        )
        self._runtime = _attest_bridge_runtime(self.root, python_path)
        if (
            expected_runtime is not None
            and self._runtime.attestation_sha256 != expected_runtime.attestation_sha256
        ):
            raise ReadinessBlocked(
                "bridge interpreter does not match the audited runtime attestation"
            )

    @property
    def artifact(self) -> InputArtifact:
        return self._artifact

    @property
    def runtime(self) -> BridgeRuntimeAttestation:
        return self._runtime

    def exchange(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if sha256_file(self.path) != self._artifact.sha256:
            raise ReadinessBlocked("external bridge changed after readiness audit")
        current_runtime = _attest_bridge_runtime(self.root, self._runtime.python_path)
        if current_runtime.attestation_sha256 != self._runtime.attestation_sha256:
            raise ReadinessBlocked("bridge interpreter changed after readiness audit")
        payload = canonical_json_bytes(dict(request))
        environment = {
            ROOT_ENVIRONMENT_VARIABLE: str(self.root),
            "PYTHONHASHSEED": str(SEED),
            "PYTHONNOUSERSITE": "1",
            "LC_ALL": "C.UTF-8",
        }
        try:
            completed = subprocess.run(
                (
                    self._runtime.python_path,
                    *BRIDGE_INVOCATION_FLAGS,
                    str(self.path),
                ),
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.root,
                env=environment,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BridgeProtocolError("external bridge process failed") from exc
        if completed.returncode != 0:
            raise BridgeProtocolError(
                f"external bridge exited with status {completed.returncode}; stderr suppressed"
            )
        if len(completed.stdout) > MAX_BRIDGE_RESPONSE_BYTES:
            raise BridgeProtocolError("external bridge response exceeded size limit")
        try:
            response = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeProtocolError("external bridge returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise BridgeProtocolError("external bridge response must be an object")
        return response


def _validate_capabilities(
    value: Mapping[str, Any], *, expected_python_version: str
) -> None:
    if value.get("protocol") != BRIDGE_PROTOCOL:
        raise BridgeProtocolError("bridge protocol identifier mismatch")
    if value.get("upstream_commit") != PINNED_COMMIT:
        raise BridgeProtocolError("bridge does not attest the pinned upstream commit")
    if value.get("contract_sha256") != BRIDGE_CONTRACT_SHA256:
        raise BridgeProtocolError("bridge contract hash mismatch")
    if value.get("transport_mode") != "emit_requests_only":
        raise BridgeProtocolError("bridge must emit requests and make no provider calls")
    if tuple(value.get("stages", ())) != STAGES:
        raise BridgeProtocolError("bridge stage order does not match the upstream pipeline")
    renderer = value.get("renderer")
    if not isinstance(renderer, Mapping) or renderer.get("kind") != "rule_based":
        raise BridgeProtocolError("bridge must expose the upstream rule-based renderer")
    runtime = value.get("runtime")
    dependencies = runtime.get("dependencies") if isinstance(runtime, Mapping) else None
    if not isinstance(dependencies, Mapping):
        raise BridgeProtocolError("bridge did not attest its runtime dependencies")
    if runtime.get("python") != expected_python_version:
        raise BridgeProtocolError(
            "bridge Python version does not match the frozen interpreter attestation"
        )
    resolved = {
        str(name): str(details.get("version"))
        for name, details in dependencies.items()
        if isinstance(details, Mapping) and details.get("available") is True
    }
    if resolved != dict(BRIDGE_RUNTIME_DEPENDENCIES):
        raise BridgeProtocolError(
            "bridge runtime does not match evolving_intent_math_verify.lock"
        )
    if runtime.get("math_verifier_mode") != "math_verify":
        raise BridgeProtocolError("bridge would use the non-fidelity math fallback")
    if value.get("compatibility_patches") != list(BRIDGE_COMPATIBILITY_PATCHES):
        raise BridgeProtocolError(
            "bridge did not attest the pinned upstream placeholder repair"
        )


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        existing = read_json(path)
        if sha256_json(existing) != sha256_json(dict(value)):
            raise ResumeBlocked(f"immutable build artifact differs: {path}")
        return sha256_file(path)
    return atomic_write_json(path, dict(value))


def _prompt_file_artifacts(root: Path, values: Any) -> tuple[InputArtifact, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise BridgeProtocolError("each model call must declare its upstream prompt_files")
    artifacts: list[InputArtifact] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise BridgeProtocolError("prompt_files entries must be non-empty strings")
        candidate = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        if not candidate.is_file() or not _inside(root, candidate):
            raise BridgeProtocolError("declared prompt file is outside the external checkout")
        artifacts.append(InputArtifact("upstream_prompt_used", str(candidate), sha256_file(candidate)))
    return tuple(artifacts)


def _call_spec(root: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeProtocolError("needs_model_call response has no call object")
    call_key = value.get("call_key")
    role = value.get("role")
    messages = value.get("messages")
    max_output_tokens = value.get("max_output_tokens")
    temperature = value.get("temperature")
    if not isinstance(call_key, str) or not _SAFE_CALL_KEY_RE.fullmatch(call_key):
        raise BridgeProtocolError("bridge call_key is invalid")
    if role not in {"generator", "judge"}:
        raise BridgeProtocolError("bridge call role must be generator or judge")
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise BridgeProtocolError("bridge call messages must be non-empty")
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise BridgeProtocolError("bridge messages must be objects")
        message_role = message.get("role")
        content = message.get("content")
        if message_role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise BridgeProtocolError("bridge messages require a valid role and string content")
        normalized_messages.append({"role": message_role, "content": content})
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise BridgeProtocolError("bridge max_output_tokens must be positive")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise BridgeProtocolError("bridge temperature must be between 0 and 2")
    schema = value.get("output_schema")
    if schema is not None and not isinstance(schema, Mapping):
        raise BridgeProtocolError("bridge output_schema must be an object or null")
    prompts = _prompt_file_artifacts(root, value.get("prompt_files"))
    return {
        "call_key": call_key,
        "role": role,
        "messages": normalized_messages,
        "max_output_tokens": max_output_tokens,
        "temperature": None if temperature is None else float(temperature),
        "output_schema": None if schema is None else dict(schema),
        "prompt_files": [
            {"role": item.role, "path": item.path, "sha256": item.sha256}
            for item in prompts
        ],
        "prompt_sha256": sha256_json(normalized_messages),
    }


def _attempt_ids(result: CompletionResult) -> list[str]:
    return [attempt.event_id for attempt in result.attempts]


class EvolvingReproductionBuilder:
    """Drive upstream semantics while retaining provider control and resumability."""

    def __init__(
        self,
        *,
        readiness: ReproductionReadiness,
        gsm8k_test_path: str | Path,
        source_ids: Sequence[int],
        output_dir: str | Path,
        settings: BuildSettings,
        transport: GenerationTransport,
        bridge: UpstreamBridge,
    ) -> None:
        if (
            not readiness.ready
            or readiness.root is None
            or readiness.bridge_runtime is None
        ):
            blockers = [issue.code for issue in readiness.issues if issue.blocking]
            raise ReadinessBlocked(f"reproduction readiness blockers: {blockers}")
        self.readiness = readiness
        self.root = Path(readiness.root)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.settings = settings
        self.transport = transport
        self.bridge = bridge
        if bridge.artifact.sha256 != sha256_file(bridge.artifact.path):
            raise ReadinessBlocked("bridge artifact hash is stale")
        bridge_runtime = getattr(bridge, "runtime", None)
        if (
            not isinstance(bridge_runtime, BridgeRuntimeAttestation)
            or bridge_runtime.attestation_sha256
            != readiness.bridge_runtime.attestation_sha256
        ):
            raise ReadinessBlocked(
                "bridge object does not match the audited interpreter runtime"
            )
        if (
            readiness.bridge_path is None
            or Path(readiness.bridge_path).resolve() != Path(bridge.artifact.path).resolve()
            or not any(
                item.path == bridge.artifact.path and item.sha256 == bridge.artifact.sha256
                for item in readiness.input_artifacts
            )
        ):
            raise ReadinessBlocked("bridge object does not match the audited bridge artifact")
        official_ids = self.root / OFFICIAL_IDS_PATH
        self.tasks, source_artifacts = select_source_tasks(
            gsm8k_test_path, official_ids, source_ids
        )
        self.source_artifacts = source_artifacts
        self._locked_artifacts = tuple(
            {
                item.path: item
                for item in (*readiness.input_artifacts, *source_artifacts)
            }.values()
        )
        self._build_config = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "benchmark": DOMAIN,
            "upstream_commit": PINNED_COMMIT,
            "seed": SEED,
            "source_ids": [task.source_id for task in self.tasks],
            "task_ids": [task.task_id for task in self.tasks],
            "generation": settings.as_dict(),
            "provider_compatibility": PROVIDER_COMPATIBILITY,
            "bridge": {
                "path": bridge.artifact.path,
                "sha256": bridge.artifact.sha256,
                "protocol": BRIDGE_PROTOCOL,
                "contract_sha256": BRIDGE_CONTRACT_SHA256,
                "runtime": readiness.bridge_runtime.as_dict(),
                "invocation": {
                    "flags": list(BRIDGE_INVOCATION_FLAGS),
                    "python_hash_seed": SEED,
                    "no_user_site": True,
                },
            },
            "inputs": [
                {"role": item.role, "path": item.path, "sha256": item.sha256}
                for item in (*readiness.input_artifacts, *source_artifacts)
            ],
            "prompt_files": [
                {"role": item.role, "path": item.path, "sha256": item.sha256}
                for item in readiness.prompt_artifacts
            ],
            "shared_across_target_arms_and_models": True,
            "target_arm": None,
            "target_model": None,
        }
        self.build_sha256 = sha256_json(self._build_config)
        self.request_scope = f"evolving-{self.build_sha256[:20]}"

    def _verify_locked_inputs(self) -> None:
        if _checkout_head(self.root) != PINNED_COMMIT:
            raise ReadinessBlocked("external checkout revision changed after readiness audit")
        for artifact in self._locked_artifacts:
            path = Path(artifact.path)
            if not path.is_file() or sha256_file(path) != artifact.sha256:
                raise ReadinessBlocked(f"locked input changed after readiness audit: {path}")

    def _verify_call_prompts(self, spec: Mapping[str, Any]) -> None:
        locked = {item.path: item.sha256 for item in self._locked_artifacts}
        for prompt in spec["prompt_files"]:
            if locked.get(prompt["path"]) != prompt["sha256"]:
                raise ReadinessBlocked(
                    "bridge declared a prompt that was not locked by the readiness audit: "
                    f"{prompt['path']}"
                )

    async def build(self, *, authorize_model_calls: bool) -> BuildResult:
        if not authorize_model_calls:
            raise ReadinessBlocked("model generation requires explicit authorization")
        self._verify_locked_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_once(self.output_dir / "build_config.json", self._build_config)
        atomic_write_json(self.output_dir / "readiness.json", self.readiness.as_dict())

        capabilities = self.bridge.exchange(
            {
                "protocol": BRIDGE_PROTOCOL,
                "operation": "capabilities",
                "contract_sha256": BRIDGE_CONTRACT_SHA256,
                "seed": SEED,
            }
        )
        _validate_capabilities(
            capabilities,
            expected_python_version=self.readiness.bridge_runtime.python_version,
        )
        _write_once(
            self.output_dir / "bridge_capabilities.json",
            {"capabilities": dict(capabilities), "sha256": sha256_json(capabilities)},
        )

        rendered_paths: list[Path] = []
        for task in self.tasks:
            self._verify_locked_inputs()
            rendered_paths.append(await self._build_task(task))
        final_path = self.output_dir / "evolving_intent_gsm8k_frozen.json"
        final_payload = self._combine_rendered(rendered_paths)
        _write_once(final_path, final_payload)
        # The public adapter is the final independent format/redaction check.
        adapter = EvolvingIntentAdapter(final_path, expected_sha256=sha256_file(final_path))
        loaded = adapter.load_tasks()
        if len(loaded) != 2 * len(self.tasks):
            raise ReproductionError("final adapter validation returned the wrong task count")

        receipt_payload = self._build_receipt(final_path, rendered_paths)
        receipt_path = self.output_dir / "build_receipt.json"
        atomic_write_json(receipt_path, receipt_payload)
        return BuildResult(
            dataset_path=str(final_path),
            dataset_sha256=sha256_file(final_path),
            receipt_path=str(receipt_path),
            receipt_sha256=sha256_file(receipt_path),
            num_source_tasks=len(self.tasks),
            num_condition_records=len(loaded),
        )

    async def _build_task(self, task: SourceTask) -> Path:
        task_dir = self.output_dir / "tasks" / str(task.source_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        source_payload = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "build_sha256": self.build_sha256,
            "source": task.private_dict(),
            "source_sha256": sha256_json(task.private_dict()),
        }
        _write_once(task_dir / "source.json", source_payload)

        prior: dict[str, Any] = {}
        for stage in STAGES:
            artifact = await self._run_stage(task, task_dir, stage, prior)
            prior[stage] = artifact
        return self._render_pair(task, task_dir, prior)

    async def _run_stage(
        self,
        task: SourceTask,
        task_dir: Path,
        stage: str,
        prior_artifacts: Mapping[str, Any],
    ) -> Any:
        stage_path = task_dir / f"{stage}.json"
        work_path = task_dir / f"{stage}.work.json"
        input_sha256 = sha256_json(
            {
                "build_sha256": self.build_sha256,
                "stage": stage,
                "source": task.private_dict(),
                "prior_artifacts": dict(prior_artifacts),
            }
        )
        if stage_path.exists():
            completed = read_json(stage_path)
            if (
                not isinstance(completed, Mapping)
                or completed.get("status") != "complete"
                or completed.get("input_sha256") != input_sha256
                or completed.get("artifact_sha256")
                != sha256_json(completed.get("artifact"))
            ):
                raise ResumeBlocked(f"completed stage is incompatible or corrupt: {stage_path}")
            return completed["artifact"]

        state: Any = None
        model_result: Any = None
        calls: list[dict[str, Any]] = []
        if work_path.exists():
            work = read_json(work_path)
            if not isinstance(work, Mapping) or work.get("input_sha256") != input_sha256:
                raise ResumeBlocked(f"stage work does not match current inputs: {work_path}")
            status = work.get("status")
            if status == "call_pending":
                response_path_value = work.get("response_path")
                response_path = (
                    Path(response_path_value)
                    if isinstance(response_path_value, str)
                    else None
                )
                recoverable = (
                    response_path is not None
                    and response_path.is_file()
                    and response_path.resolve().is_relative_to(task_dir.resolve())
                )
                if recoverable:
                    saved_response = read_json(response_path)
                    if (
                        not isinstance(saved_response, Mapping)
                        or saved_response.get("request_artifact_sha256")
                        != work.get("request_artifact_sha256")
                        or not isinstance(saved_response.get("receipt"), Mapping)
                        or not isinstance(saved_response.get("model_result"), Mapping)
                    ):
                        raise ResumeBlocked(f"saved call response is incompatible: {response_path}")
                    response_sha256 = sha256_file(response_path)
                    recovered_receipt = dict(saved_response["receipt"])
                    recovered_receipt["request_artifact"] = {
                        "path": work.get("request_path"),
                        "sha256": work.get("request_artifact_sha256"),
                    }
                    recovered_receipt["response_artifact"] = {
                        "path": str(response_path),
                        "sha256": response_sha256,
                    }
                    state = work.get("bridge_state")
                    model_result = dict(saved_response["model_result"])
                    calls = [*work.get("calls", ()), recovered_receipt]
                    atomic_write_json(
                        work_path,
                        {
                            "schema_version": BUILD_SCHEMA_VERSION,
                            "input_sha256": input_sha256,
                            "status": "result_ready",
                            "bridge_state": state,
                            "model_result": model_result,
                            "calls": calls,
                        },
                    )
                else:
                    raise ResumeBlocked(
                        f"{work_path} contains an unresolved provider call; inspect call events "
                        "instead of silently issuing a duplicate"
                    )
            elif status == "call_failed":
                raise ResumeBlocked(
                    f"{work_path} contains an unresolved provider call; inspect call events "
                    "instead of silently issuing a duplicate"
                )
            elif status not in {"ready", "result_ready"}:
                raise ResumeBlocked(f"unrecognized resumable stage state: {status!r}")
            if status != "call_pending":
                state = work.get("bridge_state")
                model_result = work.get("model_result")
                calls = list(work.get("calls", ()))

        for step in range(MAX_BRIDGE_STEPS_PER_STAGE):
            request = {
                "protocol": BRIDGE_PROTOCOL,
                "operation": "advance_stage",
                "contract_sha256": BRIDGE_CONTRACT_SHA256,
                "upstream_commit": PINNED_COMMIT,
                "seed": SEED,
                "stage": stage,
                "source_task": task.private_dict(),
                "prior_artifacts": dict(prior_artifacts),
                "state": state,
                "model_result": model_result,
            }
            response = self.bridge.exchange(request)
            response_sha256 = sha256_json(response)
            status = response.get("status")
            if status == "complete":
                if "artifact" not in response:
                    raise BridgeProtocolError("complete stage response has no artifact")
                artifact = response["artifact"]
                artifact_sha256 = sha256_json(artifact)
                completed_payload = {
                    "schema_version": BUILD_SCHEMA_VERSION,
                    "build_sha256": self.build_sha256,
                    "stage": stage,
                    "status": "complete",
                    "input_sha256": input_sha256,
                    "artifact": artifact,
                    "artifact_sha256": artifact_sha256,
                    "calls": calls,
                    "final_bridge_response_sha256": response_sha256,
                }
                atomic_write_json(stage_path, completed_payload)
                atomic_write_json(
                    work_path,
                    {
                        "schema_version": BUILD_SCHEMA_VERSION,
                        "input_sha256": input_sha256,
                        "status": "complete",
                        "stage_artifact_sha256": sha256_file(stage_path),
                    },
                )
                return artifact
            if status != "needs_model_call":
                raise BridgeProtocolError(f"unexpected advance_stage status: {status!r}")

            spec = _call_spec(self.root, response.get("call"))
            self._verify_call_prompts(spec)
            if any(call.get("call_key") == spec["call_key"] for call in calls):
                raise BridgeProtocolError(f"bridge repeated call_key {spec['call_key']!r}")
            state = response.get("state")
            call_number = len(calls) + 1
            call_stem = f"{call_number:03d}-{sha256_bytes(spec['call_key'].encode('utf-8'))[:12]}"
            call_dir = task_dir / "calls" / stage
            request_path = call_dir / f"{call_stem}.request.json"
            response_path = call_dir / f"{call_stem}.response.json"
            request_artifact = {
                "schema_version": BUILD_SCHEMA_VERSION,
                "build_sha256": self.build_sha256,
                "source_id": task.source_id,
                "task_id": task.task_id,
                "stage": stage,
                "call": spec,
                "model": self._model_for_role(spec["role"]),
                "reasoning_effort": self._reasoning_for_role(spec["role"]),
                "upstream_max_output_tokens": spec["max_output_tokens"],
                "provider_max_output_tokens": self._effective_max_output_tokens(
                    spec["role"], spec["max_output_tokens"]
                ),
            }
            request_artifact_sha256 = _write_once(request_path, request_artifact)
            pending = {
                "schema_version": BUILD_SCHEMA_VERSION,
                "input_sha256": input_sha256,
                "status": "call_pending",
                "bridge_state": state,
                "bridge_response_sha256": response_sha256,
                "call": spec,
                "request_path": str(request_path),
                "request_artifact_sha256": request_artifact_sha256,
                "response_path": str(response_path),
                "model_result": None,
                "calls": calls,
            }
            atomic_write_json(work_path, pending)
            try:
                result = await self._model_call(task, stage, spec)
            except TransportError as exc:
                pending["status"] = "call_failed"
                pending["transport_error"] = {
                    "category": exc.category,
                    "attempt_event_ids": [attempt.event_id for attempt in exc.attempts],
                }
                atomic_write_json(work_path, pending)
                raise
            base_receipt = {
                "call_key": spec["call_key"],
                "role": spec["role"],
                "model": self._model_for_role(spec["role"]),
                "resolved_model_id": result.model_id,
                "settings": {
                    "temperature": spec["temperature"],
                    "reasoning_effort": self._reasoning_for_role(spec["role"]),
                    "upstream_max_output_tokens": spec["max_output_tokens"],
                    "provider_max_output_tokens": self._effective_max_output_tokens(
                        spec["role"], spec["max_output_tokens"]
                    ),
                    "output_schema_sha256": (
                        None
                        if spec["output_schema"] is None
                        else sha256_json(spec["output_schema"])
                    ),
                },
                "prompt_sha256": spec["prompt_sha256"],
                "prompt_files": spec["prompt_files"],
                "output_sha256": sha256_bytes(result.text.encode("utf-8")),
                "response_id": result.response_id,
                "request_id": result.request_id,
                "attempt_event_ids": _attempt_ids(result),
                "finish_reason": result.finish_reason,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "cached_input_tokens": result.usage.cached_input_tokens,
                    "reasoning_tokens": result.usage.reasoning_tokens,
                },
                "accounted_cost_usd": str(result.cost_usd),
            }
            model_result = {
                "call_key": spec["call_key"],
                "text": result.text,
                "output_sha256": base_receipt["output_sha256"],
                "response_id": result.response_id,
                "request_id": result.request_id,
                "attempt_event_ids": base_receipt["attempt_event_ids"],
            }
            response_artifact = {
                "schema_version": BUILD_SCHEMA_VERSION,
                "build_sha256": self.build_sha256,
                "request_artifact_sha256": request_artifact_sha256,
                "model_result": model_result,
                "receipt": base_receipt,
            }
            response_artifact_sha256 = _write_once(response_path, response_artifact)
            receipt = {
                **base_receipt,
                "request_artifact": {
                    "path": str(request_path),
                    "sha256": request_artifact_sha256,
                },
                "response_artifact": {
                    "path": str(response_path),
                    "sha256": response_artifact_sha256,
                },
            }
            calls.append(receipt)
            atomic_write_json(
                work_path,
                {
                    "schema_version": BUILD_SCHEMA_VERSION,
                    "input_sha256": input_sha256,
                    "status": "result_ready",
                    "bridge_state": state,
                    "model_result": model_result,
                    "calls": calls,
                },
            )
        raise BridgeProtocolError(f"stage exceeded {MAX_BRIDGE_STEPS_PER_STAGE} bridge steps")

    def _model_for_role(self, role: str) -> str:
        return self.settings.generator_model if role == "generator" else self.settings.judge_model

    def _reasoning_for_role(self, role: str) -> str | None:
        return (
            self.settings.generator_reasoning_effort
            if role == "generator"
            else self.settings.judge_reasoning_effort
        )

    def _effective_max_output_tokens(self, role: str, upstream_limit: int) -> int:
        """Adapt only a provider's syntactic minimum, retaining both limits."""

        model_name = self._model_for_role(role)
        model_spec = CATALOG.models.get(model_name)
        if model_spec is not None and model_spec.provider == "openai":
            return max(PROVIDER_COMPATIBILITY["openai_min_output_tokens"], upstream_limit)
        return upstream_limit

    async def _model_call(
        self,
        task: SourceTask,
        stage: str,
        spec: Mapping[str, Any],
    ) -> CompletionResult:
        schema = spec["output_schema"]
        output_schema = (
            None
            if schema is None
            else JsonSchemaOutput.from_schema(
                "evolving_stage_output",
                schema,
                description="Upstream Evolving Intent stage output",
            )
        )
        call_hash = sha256_json(
            {"build_sha256": self.build_sha256, "task": task.source_id, "call": spec["call_key"]}
        )[:20]
        input_token_estimate = 256 + sum(
            len(message["content"].encode("utf-8")) for message in spec["messages"]
        )
        return await self.transport.complete(
            self._model_for_role(spec["role"]),
            spec["messages"],
            purpose=f"evolving_{stage}",
            request_key=f"{self.request_scope}/{task.source_id}/{stage}/{call_hash}",
            input_token_estimate=input_token_estimate,
            max_output_tokens=self._effective_max_output_tokens(
                spec["role"], spec["max_output_tokens"]
            ),
            temperature=spec["temperature"],
            reasoning_effort=self._reasoning_for_role(spec["role"]),
            output_schema=output_schema,
        )

    def _render_pair(
        self,
        task: SourceTask,
        task_dir: Path,
        artifacts: Mapping[str, Any],
    ) -> Path:
        path = task_dir / "rendered_pair.json"
        input_sha256 = sha256_json(
            {
                "build_sha256": self.build_sha256,
                "source": task.private_dict(),
                "artifacts": dict(artifacts),
                "render": {
                    "seed": SEED,
                    "t1": {"num_turns": 1, "num_revisions": 0, "num_switches": 0},
                    "t7": {"num_turns": 7, "num_revisions": 2, "num_switches": 2},
                },
            }
        )
        if path.exists():
            value = read_json(path)
            if not isinstance(value, Mapping) or value.get("input_sha256") != input_sha256:
                raise ResumeBlocked(f"rendered pair does not match current inputs: {path}")
            self._validate_rendered_file(path, task.task_id)
            return path

        response = self.bridge.exchange(
            {
                "protocol": BRIDGE_PROTOCOL,
                "operation": "render_pair",
                "contract_sha256": BRIDGE_CONTRACT_SHA256,
                "upstream_commit": PINNED_COMMIT,
                "seed": SEED,
                "source_task": task.private_dict(),
                "stage_artifacts": dict(artifacts),
                "conditions": {
                    "t1": {"num_turns": 1, "num_revisions": 0, "num_switches": 0},
                    "t7": {"num_turns": 7, "num_revisions": 2, "num_switches": 2},
                },
            }
        )
        if response.get("status") != "rendered":
            raise BridgeProtocolError("render_pair did not return rendered status")
        simulator = response.get("simulator")
        if not isinstance(simulator, Mapping) or simulator.get("kind") != "rule_based":
            raise BridgeProtocolError("render_pair did not attest the rule-based simulator")
        records = self._safe_rendered_records(response.get("records"), task.task_id)
        payload = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "input_sha256": input_sha256,
            "bridge_response_sha256": sha256_json(response),
            "simulator": {
                "kind": "rule_based",
                "seed": SEED,
                "t1": {"num_turns": 1, "num_revisions": 0, "num_switches": 0},
                "t7": {"num_turns": 7, "num_revisions": 2, "num_switches": 2},
            },
            "tasks": records,
        }
        atomic_write_json(path, payload)
        self._validate_rendered_file(path, task.task_id)
        return path

    @staticmethod
    def _safe_rendered_records(value: Any, expected_task_id: str) -> list[dict[str, Any]]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise BridgeProtocolError("rendered records must be an array")
        safe: dict[str, dict[str, Any]] = {}
        for record in value:
            if not isinstance(record, Mapping):
                raise BridgeProtocolError("rendered record must be an object")
            if str(record.get("task_id")) != expected_task_id:
                raise BridgeProtocolError("renderer changed the official task ID")
            condition = record.get("condition")
            expected_turns = 1 if condition == "t1" else 7 if condition == "t7" else None
            turns = record.get("turns")
            label = record.get("label")
            if expected_turns is None or condition in safe:
                raise BridgeProtocolError("renderer must return one t1 and one t7 record")
            if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
                raise BridgeProtocolError("rendered turns must be an array")
            if len(turns) != expected_turns or any(
                not isinstance(turn, str) or not turn.strip() for turn in turns
            ):
                raise BridgeProtocolError(f"rendered {condition} has invalid turns")
            if isinstance(label, bool) or not isinstance(label, (str, int)) or not str(label).strip():
                raise BridgeProtocolError("rendered label must be a non-empty string or integer")
            # Reconstruct from an allowlist: latent bridge state cannot enter the public artifact.
            safe[condition] = {
                "task_id": expected_task_id,
                "condition": condition,
                "turns": list(turns),
                "label": str(label).strip(),
            }
        if set(safe) != {"t1", "t7"}:
            raise BridgeProtocolError("renderer must return paired t1 and t7 records")
        return [safe["t1"], safe["t7"]]

    @staticmethod
    def _validate_rendered_file(path: Path, expected_task_id: str) -> None:
        tasks = EvolvingIntentAdapter(path).load_tasks()
        if len(tasks) != 2 or {task.task_id for task in tasks} != {expected_task_id}:
            raise ReproductionError(f"rendered pair failed public adapter validation: {path}")

    def _combine_rendered(self, paths: Sequence[Path]) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for path in paths:
            value = read_json(path)
            records.extend(value["tasks"])
        return {
            "schema_version": BUILD_SCHEMA_VERSION,
            "benchmark": DOMAIN,
            "upstream_commit": PINNED_COMMIT,
            "seed": SEED,
            "shared_across_target_arms_and_models": True,
            "tasks": records,
        }

    def _build_receipt(self, final_path: Path, rendered_paths: Sequence[Path]) -> dict[str, Any]:
        task_receipts: list[dict[str, Any]] = []
        all_calls: list[dict[str, Any]] = []
        for task, rendered_path in zip(self.tasks, rendered_paths, strict=True):
            stage_files: list[dict[str, str]] = []
            for stage in STAGES:
                path = rendered_path.parent / f"{stage}.json"
                stage_value = read_json(path)
                stage_files.append({"stage": stage, "path": str(path), "sha256": sha256_file(path)})
                all_calls.extend(stage_value.get("calls", ()))
            task_receipts.append(
                {
                    "source_id": task.source_id,
                    "task_id": task.task_id,
                    "source_path": str(rendered_path.parent / "source.json"),
                    "source_sha256": sha256_file(rendered_path.parent / "source.json"),
                    "stage_files": stage_files,
                    "rendered_path": str(rendered_path),
                    "rendered_sha256": sha256_file(rendered_path),
                }
            )
        return {
            "schema_version": BUILD_SCHEMA_VERSION,
            "benchmark": DOMAIN,
            "build_sha256": self.build_sha256,
            "upstream_commit": PINNED_COMMIT,
            "seed": SEED,
            "frozen_dataset": {"path": str(final_path), "sha256": sha256_file(final_path)},
            "shared_across_target_arms_and_models": True,
            "generation": self.settings.as_dict(),
            "bridge_runtime": self.readiness.bridge_runtime.as_dict(),
            "tasks": task_receipts,
            "calls": all_calls,
            "prompt_files": [
                {"role": item.role, "path": item.path, "sha256": item.sha256}
                for item in self.readiness.prompt_artifacts
            ],
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit", action="store_true", help="audit only; never call a model")
    action.add_argument("--build", action="store_true", help="run/resume the reproduction")
    parser.add_argument("--gsm8k-test", required=True, help="official GSM8K test JSONL")
    parser.add_argument("--bridge", help="checkout-local transport bridge script")
    parser.add_argument(
        "--bridge-python",
        help=(
            "checkout-local virtual-environment interpreter; defaults to "
            "EVOLVING_INTENT_ROOT/.venv/bin/python"
        ),
    )
    parser.add_argument("--ids", required=True, help="zero-based source IDs, e.g. 12,14,16")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generator-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--generator-reasoning-effort")
    parser.add_argument("--judge-reasoning-effort")
    parser.add_argument("--env-file", default=str(REPOSITORY_ROOT / ".env"))
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument(
        "--stage",
        choices=(Stage.SMOKE.value, Stage.CALIBRATION.value),
        default=Stage.SMOKE.value,
        help="scope every dataset-generation reservation to this stage cap",
    )
    parser.add_argument("--yes-spend", action="store_true")
    return parser


async def _run_build(args: argparse.Namespace, readiness: ReproductionReadiness) -> int:
    if not args.yes_spend:
        raise ReadinessBlocked("--build requires --yes-spend")
    if not args.generator_model or not args.judge_model:
        raise ValueError("--build requires --generator-model and --judge-model")
    if (
        readiness.root is None
        or readiness.bridge_path is None
        or readiness.bridge_runtime is None
    ):
        raise ReadinessBlocked("external root, bridge, or bridge runtime unavailable")
    output_dir = Path(args.output_dir).expanduser().resolve()
    ledger = BudgetLedger(
        Path(args.artifacts) / "_global_budget.sqlite3",
        operational_caps_usd={
            provider: Decimal(str(amount))
            for provider, amount in OPERATIONAL_PROVIDER_USD.items()
        },
    )
    transport = Transport(
        ledger,
        output_dir / "call_attempts.jsonl",
        environ=_environment(args.env_file),
    )
    bridge = SubprocessUpstreamBridge(
        readiness.root,
        readiness.bridge_path,
        readiness.bridge_runtime.python_path,
        expected_runtime=readiness.bridge_runtime,
    )
    builder = EvolvingReproductionBuilder(
        readiness=readiness,
        gsm8k_test_path=args.gsm8k_test,
        source_ids=parse_ids(args.ids),
        output_dir=output_dir,
        settings=BuildSettings(
            generator_model=args.generator_model,
            judge_model=args.judge_model,
            generator_reasoning_effort=args.generator_reasoning_effort,
            judge_reasoning_effort=args.judge_reasoning_effort,
        ),
        transport=transport,
        bridge=bridge,
    )
    # The build hash is needed for a collision-free request scope, so install
    # the stricter scoped ledger only after the immutable builder is formed.
    stage = Stage(args.stage)
    transport.ledger = BudgetLedger(
        Path(args.artifacts) / "_global_budget.sqlite3",
        operational_caps_usd={
            provider: Decimal(str(amount))
            for provider, amount in OPERATIONAL_PROVIDER_USD.items()
        },
        request_scope=builder.request_scope,
        scope_caps_usd={
            provider: Decimal(str(amount))
            for provider, amount in STAGE_PROVIDER_USD[stage].items()
        },
    )
    _write_once(
        output_dir / "spend_scope.json",
        {
            "request_scope": builder.request_scope,
            "stage": stage.value,
            "scope_caps_usd": STAGE_PROVIDER_USD[stage],
            "operational_caps_usd": OPERATIONAL_PROVIDER_USD,
        },
    )
    result = await builder.build(authorize_model_calls=True)
    print(f"frozen dataset: {result.dataset_path}")
    print(f"sha256: {result.dataset_sha256}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        source_ids = parse_ids(args.ids)
        readiness = audit_reproduction(
            gsm8k_test_path=args.gsm8k_test,
            bridge_path=args.bridge,
            bridge_python_path=args.bridge_python,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "readiness.json", readiness.as_dict())
        if args.audit:
            print("READY" if readiness.ready else "BLOCKED")
            for issue in readiness.issues:
                print(f"{'BLOCKER' if issue.blocking else 'NOTE'} {issue.code}: {issue.detail}")
            print(f"selected source IDs: {','.join(str(item) for item in source_ids)}")
            return 0 if readiness.ready else 2
        if not readiness.ready:
            raise ReadinessBlocked(
                ", ".join(issue.code for issue in readiness.issues if issue.blocking)
            )
        return asyncio.run(_run_build(args, readiness))
    except (OSError, ValueError, ReproductionError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRIDGE_CONTRACT",
    "BRIDGE_CONTRACT_SHA256",
    "BRIDGE_PYTHON_MAJOR_MINOR",
    "BRIDGE_PROTOCOL",
    "BridgeRuntimeAttestation",
    "BridgeRuntimeError",
    "BuildResult",
    "BuildSettings",
    "BridgeProtocolError",
    "EvolvingReproductionBuilder",
    "OfficialTaskID",
    "ReadinessBlocked",
    "ReproductionReadiness",
    "ResumeBlocked",
    "SEED",
    "STAGES",
    "SourceTask",
    "SubprocessUpstreamBridge",
    "audit_reproduction",
    "load_official_id_map",
    "parse_ids",
    "select_source_tasks",
]
