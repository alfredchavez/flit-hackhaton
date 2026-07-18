"""Coarse beat-synchronous chord timeline (major/minor templates)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np


_MAJOR = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
_MINOR = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
_PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_PC_MINOR = ["Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "Abm", "Am", "Bbm", "Bm"]
_PC_MAJOR = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _templates() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for i in range(12):
        out.append((_PC_MAJOR[i], np.roll(_MAJOR, i)))
        out.append((_PC_MINOR[i], np.roll(_MINOR, i)))
    return out


def estimate_chords(
    aligned_wav: Path,
    beats: list[float],
    *,
    duration_seconds: float | None = None,
    min_confidence: float = 0.12,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Coarse chord estimation: beat-synchronous chroma + major/minor templates.

    Public stage entrypoint.
    Returns (chords, warnings). chords may be empty with chords_unreliable warning.
    """
    aligned_wav = Path(aligned_wav)
    y, sr = librosa.load(str(aligned_wav), sr=None, mono=True)
    if duration_seconds is None:
        duration_seconds = float(len(y) / sr)

    warnings: list[str] = []
    if len(beats) < 2:
        warnings.append("chords_unreliable")
        return [], warnings

    # Ensure beat grid covers the track.
    beat_times = list(beats)
    if beat_times[0] > 0.05:
        beat_times = [0.0] + beat_times
    if beat_times[-1] < duration_seconds - 0.1:
        beat_times = beat_times + [duration_seconds]

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    templates = _templates()

    raw: list[dict[str, Any]] = []
    confidences: list[float] = []
    for i in range(len(beat_times) - 1):
        start, end = float(beat_times[i]), float(beat_times[i + 1])
        if end <= start:
            continue
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            # nearest frame
            idx = int(np.argmin(np.abs(times - start)))
            vec = chroma[:, idx]
        else:
            vec = np.mean(chroma[:, mask], axis=1)
        if np.allclose(vec, 0):
            chord = "N"
            conf = 0.0
        else:
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            best_name = "N"
            best = -np.inf
            second = -np.inf
            for name, tmpl in templates:
                t = tmpl / (np.linalg.norm(tmpl) + 1e-9)
                score = float(np.dot(vec, t))
                if score > best:
                    second = best
                    best = score
                    best_name = name
                elif score > second:
                    second = score
            conf = max(0.0, best - second)
            chord = best_name
        raw.append({"start": start, "end": end, "chord": chord})
        confidences.append(conf)

    if not raw or float(np.mean(confidences)) < min_confidence:
        warnings.append("chords_unreliable")
        return [], warnings

    # Merge adjacent identical chords.
    merged: list[dict[str, Any]] = []
    for item in raw:
        if item["chord"] == "N":
            continue
        if merged and merged[-1]["chord"] == item["chord"]:
            merged[-1]["end"] = item["end"]
        else:
            merged.append(dict(item))

    if len(merged) < 2:
        warnings.append("chords_unreliable")
        return [], warnings

    return merged, warnings
