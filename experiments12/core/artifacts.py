"""Crash-safe, deterministic artifact I/O for Experiment 12.

Whole JSON and JSONL files are serialized before the destination is touched,
written to a temporary file in the same directory, fsynced, and atomically
installed with ``os.replace``.  ``append_jsonl`` uses an advisory process lock,
one ``O_APPEND`` write, and fsync for durable append-only event logs.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from decimal import Decimal
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator

from .schemas import record_to_dict


PathLike = str | os.PathLike[str]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return record_to_dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return format(value, "f")
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used for content-addressing and JSONL records."""

    return json.dumps(
        value,
        default=_jsonable,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str, *, encoding: str = "utf-8") -> str:
    return sha256_bytes(value.encode(encoding))


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist a rename where the platform supports directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: PathLike, payload: bytes, *, mode: int = 0o644) -> str:
    """Atomically replace ``path`` and return the SHA256 of exact file bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        fd, raw_temp = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temp_path = Path(raw_temp)
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(temp_path, destination)
        temp_path = None
        _fsync_directory(destination.parent)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    return sha256_bytes(payload)


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o644,
) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return atomic_write_bytes(path, text.encode(encoding), mode=mode)


def atomic_write_json(
    path: PathLike,
    value: Any,
    *,
    pretty: bool = True,
    mode: int = 0o644,
) -> str:
    """Serialize first, then atomically replace a JSON artifact."""

    if pretty:
        payload = (
            json.dumps(
                value,
                default=_jsonable,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    else:
        payload = canonical_json_bytes(value) + b"\n"
    return atomic_write_bytes(path, payload, mode=mode)


def _jsonl_payload(records: Iterable[Any]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def atomic_write_jsonl(path: PathLike, records: Iterable[Any], *, mode: int = 0o644) -> str:
    """Atomically replace a complete JSONL artifact."""

    return atomic_write_bytes(path, _jsonl_payload(records), mode=mode)


def append_jsonl(path: PathLike, record: Any, *, mode: int = 0o644) -> str:
    """Append one process-safe, fsynced JSONL event and return its content hash.

    The record is fully serialized before opening the destination.  A single
    ``O_APPEND`` write is protected by ``flock``, preventing interleaved lines
    from concurrent Experiment 12 workers on the same host.
    """

    line = canonical_json_bytes(record) + b"\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination, os.O_WRONLY | os.O_APPEND | os.O_CREAT, mode)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        written = os.write(fd, line)
        if written != len(line):
            raise OSError(f"short JSONL append: wrote {written} of {len(line)} bytes")
        os.fsync(fd)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    return sha256_bytes(line)


def read_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: PathLike) -> Iterator[Any]:
    """Yield JSONL records, rejecting blank or torn trailing lines."""

    with Path(path).open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"torn JSONL record at line {line_number}")
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record at line {line_number}: {exc.msg}") from exc


def read_jsonl(path: PathLike) -> list[Any]:
    return list(iter_jsonl(path))


def verify_sha256(path: PathLike, expected: str) -> bool:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in expected)
    ):
        raise ValueError("expected must be a SHA256 hex digest")
    return sha256_file(path) == expected.lower()


__all__ = [
    "canonical_json_bytes",
    "sha256_bytes",
    "sha256_text",
    "sha256_json",
    "sha256_file",
    "verify_sha256",
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "atomic_write_jsonl",
    "append_jsonl",
    "read_json",
    "iter_jsonl",
    "read_jsonl",
]
