"""Cheap stage completion markers for safe re-runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def source_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "filename": path.name,
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def marker_path(stage_dir: Path, stage_name: str) -> Path:
    return stage_dir / f".stage_{stage_name}.json"


def write_marker(
    stage_dir: Path,
    stage_name: str,
    *,
    source: Path,
    options: dict[str, Any],
    outputs: Iterable[str | Path],
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source_fingerprint(source),
        "stage": stage_name,
        "options": options,
        "outputs": [str(Path(p).name) if Path(p).parent == stage_dir else str(p) for p in outputs],
    }
    # Store relative names when under stage_dir for stability.
    resolved = []
    for p in outputs:
        p = Path(p)
        try:
            resolved.append(str(p.resolve().relative_to(stage_dir.resolve())))
        except ValueError:
            resolved.append(str(p.resolve()))
    payload["outputs"] = resolved
    marker_path(stage_dir, stage_name).write_text(json.dumps(payload, indent=2, sort_keys=True))


def stage_complete(
    stage_dir: Path,
    stage_name: str,
    *,
    source: Path,
    options: dict[str, Any],
    expected_outputs: Iterable[str | Path],
    force: bool = False,
) -> bool:
    """Return True when stage can be skipped (marker matches and all outputs exist)."""
    if force:
        return False
    marker = marker_path(stage_dir, stage_name)
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    if data.get("stage") != stage_name:
        return False
    if data.get("source") != source_fingerprint(source):
        return False
    if data.get("options") != options:
        return False

    expected = []
    for p in expected_outputs:
        p = Path(p)
        if not p.is_absolute():
            p = stage_dir / p
        expected.append(p.resolve())

    recorded = data.get("outputs") or []
    recorded_paths = []
    for r in recorded:
        rp = Path(r)
        if not rp.is_absolute():
            rp = stage_dir / rp
        recorded_paths.append(rp.resolve())

    if set(map(str, recorded_paths)) != set(map(str, expected)):
        return False

    for p in expected:
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True
