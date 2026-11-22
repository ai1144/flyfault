"""Artifact persistence and execution provenance."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import cache
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value: object) -> str:
    return sha256(canonical(value)).hexdigest()


@contextmanager
def atomic(path: Path, *, binary: bool = False):
    with NamedTemporaryFile(mode="wb" if binary else "w", dir=path.parent, delete=False) as file:
        tmp_path = Path(file.name)
        try:
            yield file
            file.close()
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)


def save_json(path: Path, value: object) -> None:
    with atomic(path) as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def save_signals(path: Path, **arrays: np.ndarray) -> None:
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Non-finite values in signal {name}")
    with atomic(path, binary=True) as file:
        np.savez_compressed(file, **arrays)


@contextmanager
def artifact_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@cache
def runtime() -> dict:
    project = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
        "packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
        "git_revision": revision,
    }


@cache
def science_digest() -> str:
    root = Path(__file__).parent
    modules = (
        "config", "environment", "sensing", "locomotion", "recording", "network",
        "readout", "interventions", "metrics", "runner",
    )
    sources = {
        name: sha256((root / f"{name}.py").read_bytes()).hexdigest()
        for name in modules if (root / f"{name}.py").exists()
    }
    return fingerprint(sources)
