from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def get_git_commit(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:
        return None


def runtime_source_identity(repo_root: Path) -> dict[str, Any]:
    """Fingerprint the exact Python implementation used by a training run."""
    root = Path(repo_root).resolve()
    paths = sorted((root / "src").rglob("*.py"))
    entrypoint = root / "scripts" / "train.py"
    if entrypoint.is_file():
        paths.append(entrypoint)
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in paths:
        relative = str(path.relative_to(root))
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(relative_paths),
        "files": relative_paths,
    }


def environment_snapshot() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
    }
