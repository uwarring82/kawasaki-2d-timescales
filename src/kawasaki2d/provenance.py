"""Provenance: per-run manifests capturing ``(config, commit, seed, environment)``.

Task-card requirement: *no result enters ``results/`` without a manifest*, and
every reported number is regenerable from the recorded tuple. This module builds
the machine-generated manifest; the human logbook (``logbook/``) is complementary.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from . import __version__
from .rng import rng_fingerprint

# Schema version for the manifest format itself (bump on breaking changes).
MANIFEST_SCHEMA_VERSION = "1.0"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict) -> str:
    """SHA-256 of the canonical JSON encoding of a config dict."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def file_checksum(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(args: list[str], cwd: Path | None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


@dataclass
class GitState:
    commit: str | None
    dirty: bool | None
    branch: str | None

    @classmethod
    def capture(cls, cwd: Path | None = None) -> "GitState":
        commit = _run_git(["rev-parse", "HEAD"], cwd)
        status = _run_git(["status", "--porcelain"], cwd)
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        dirty = None if status is None else (len(status) > 0)
        return cls(commit=commit, dirty=dirty, branch=branch)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("numpy", "scipy", "pyyaml", "PyYAML", "numba", "matplotlib", "pandas", "pyarrow"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def environment_fingerprint() -> dict[str, Any]:
    """Capture the runtime environment for reproducibility."""
    from .dynamics import numba_available

    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kawasaki2d_version": __version__,
        "numba_active": numba_available(),
        "rng": rng_fingerprint().as_dict(),
        "packages": _package_versions(),
    }


@dataclass
class Manifest:
    """The machine-generated provenance record for one run."""

    run_id: str
    timestamp: str
    config: dict
    config_hash: str
    seeds: dict[str, int]
    git: dict[str, Any]
    environment: dict[str, Any]
    outputs: dict[str, str] = field(default_factory=dict)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    notes: str = ""

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        timestamp: str,
        config: dict,
        seeds: dict[str, int],
        repo_root: Path | None = None,
        notes: str = "",
    ) -> "Manifest":
        """Assemble a manifest. ``timestamp`` is supplied by the caller (ISO-8601)
        so that manifest construction stays free of hidden clock state."""
        return cls(
            run_id=run_id,
            timestamp=timestamp,
            config=config,
            config_hash=config_hash(config),
            seeds=seeds,
            git=asdict(GitState.capture(repo_root)),
            environment=environment_fingerprint(),
            notes=notes,
        )

    def add_output(self, label: str, path: str | Path) -> None:
        """Record an output file and its checksum."""
        self.outputs[label] = file_checksum(path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
