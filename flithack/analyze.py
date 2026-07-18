"""BPM, beats, key, meter, energy, coarse sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


# Krumhansl-Schmuckler key profiles (major / minor)
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)
_PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass
class AnalysisResult:
    bpm: float
    meter: str
    key: str
    beats: list[float]
    downbeats: list[float]
    duration_seconds: float
    energy_curve: list[dict[str, float]]
    sections: list[dict[str, Any]]
    per_stem: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    key_confidence: float = 0.0


def estimate_key(y: np.ndarray, sr: int) -> tuple[str, float, list[str]]:
    """Return (key_string, confidence, warnings) via chroma + KS templates."""
    warnings: list[str] = []
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    if np.allclose(chroma_mean, 0):
        warnings.append("key_low_confidence")
        return "C major", 0.0, warnings

    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    best_score = -np.inf
    best_key = "C major"
    scores: list[float] = []
    for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        prof = profile / (np.linalg.norm(profile) + 1e-9)
        for i in range(12):
            rotated = np.roll(prof, i)
            score = float(np.dot(chroma_mean, rotated))
            scores.append(score)
            if score > best_score:
                best_score = score
                name = _PITCH_NAMES[i]
                best_key = f"{name} {mode}"

    scores_arr = np.array(scores)
    # Confidence: margin of best vs second best, crude.
    ordered = np.sort(scores_arr)
    margin = float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else 0.0
    conf = max(0.0, min(1.0, margin * 5.0))
    if conf < 0.15:
        warnings.append("key_low_confidence")
    return best_key, conf, warnings


def _bar_edges(downbeats: list[float], duration: float) -> list[tuple[float, float]]:
    if not downbeats:
        # Fake single bar spanning whole track.
        return [(0.0, duration)]
    edges = list(downbeats)
    if edges[0] > 0.01:
        edges = [0.0] + edges
    if edges[-1] < duration - 0.05:
        edges = edges + [duration]
    bars = []
    for i in range(len(edges) - 1):
        start, end = float(edges[i]), float(edges[i + 1])
        if end > start:
            bars.append((start, end))
    if not bars:
        bars = [(0.0, duration)]
    return bars


def energy_curve_and_sections(
    y: np.ndarray,
    sr: int,
    downbeats: list[float],
    duration: float,
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    bars = _bar_edges(downbeats, duration)
    rms_vals: list[float] = []
    for start, end in bars:
        s = int(start * sr)
        e = max(s + 1, int(end * sr))
        e = min(e, len(y))
        segment = y[s:e]
        if len(segment) == 0:
            rms_vals.append(0.0)
        else:
            rms_vals.append(float(np.sqrt(np.mean(segment.astype(float) ** 2))))

    peak = max(rms_vals) if rms_vals else 1.0
    if peak <= 0:
        peak = 1.0
    energy_curve = [
        {"start": float(s), "end": float(e), "value": float(v / peak)}
        for (s, e), v in zip(bars, rms_vals)
    ]

    # Smooth bar energy (3-bar moving average).
    vals = np.array([c["value"] for c in energy_curve], dtype=float)
    if len(vals) == 0:
        return energy_curve, []
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(vals, kernel, mode="same")

    # Split on large sustained changes.
    threshold = 0.18
    labels_raw: list[str] = []
    for v in smooth:
        if v < 0.33:
            labels_raw.append("low")
        elif v < 0.66:
            labels_raw.append("medium")
        else:
            labels_raw.append("high")

    # Segment by label runs, also split when adjacent smooth delta is large.
    segments: list[dict[str, Any]] = []
    seg_start_idx = 0
    for i in range(1, len(labels_raw)):
        big_jump = abs(smooth[i] - smooth[i - 1]) >= threshold
        label_change = labels_raw[i] != labels_raw[seg_start_idx]
        if label_change or big_jump:
            s = energy_curve[seg_start_idx]["start"]
            e = energy_curve[i - 1]["end"]
            # majority label in range
            chunk = labels_raw[seg_start_idx:i]
            lab = max(set(chunk), key=chunk.count)
            segments.append({"start": s, "end": e, "energy": lab, "_bars": i - seg_start_idx})
            seg_start_idx = i
    # tail
    s = energy_curve[seg_start_idx]["start"]
    e = energy_curve[-1]["end"]
    chunk = labels_raw[seg_start_idx:]
    lab = max(set(chunk), key=chunk.count)
    segments.append({"start": s, "end": e, "energy": lab, "_bars": len(chunk)})

    # Merge segments shorter than 4 bars into neighbors.
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if merged and seg["_bars"] < 4:
            merged[-1]["end"] = seg["end"]
            merged[-1]["_bars"] += seg["_bars"]
            # recompute energy label from combined span later if needed; keep previous label
        elif not merged and seg["_bars"] < 4 and len(segments) > 1:
            # merge forward by absorbing into next — stash
            if not merged:
                merged.append(seg)
            else:
                merged[-1]["end"] = seg["end"]
                merged[-1]["_bars"] += seg["_bars"]
        else:
            merged.append(seg)

    # Second pass: if first is short, merge into second.
    if len(merged) >= 2 and merged[0]["_bars"] < 4:
        merged[1]["start"] = merged[0]["start"]
        merged[1]["_bars"] += merged[0]["_bars"]
        merged = merged[1:]

    sections = [{"start": m["start"], "end": m["end"], "energy": m["energy"]} for m in merged]
    return energy_curve, sections


def compute_per_stem_stats(
    midi_paths: dict[str, Path],
    downbeats: list[float],
    duration: float,
) -> dict[str, Any]:
    """Density / register stats from MIDI (called after transcription)."""
    import pretty_midi

    n_bars = max(1, len(_bar_edges(downbeats, duration)))
    per_stem: dict[str, Any] = {}

    drums = midi_paths.get("drums")
    if drums and drums.is_file():
        pm = pretty_midi.PrettyMIDI(str(drums))
        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        per_stem["drums"] = {"onsets_per_bar": float(n_notes) / n_bars}

    bass = midi_paths.get("bass")
    if bass and bass.is_file():
        pm = pretty_midi.PrettyMIDI(str(bass))
        pitches = [n.pitch for inst in pm.instruments for n in inst.notes]
        n_notes = len(pitches)
        per_stem["bass"] = {
            "notes_per_bar": float(n_notes) / n_bars,
            "register_low_midi": int(min(pitches)) if pitches else 0,
            "register_high_midi": int(max(pitches)) if pitches else 0,
        }

    melody = midi_paths.get("melody")
    if melody and melody.is_file():
        pm = pretty_midi.PrettyMIDI(str(melody))
        pitches = [n.pitch for inst in pm.instruments for n in inst.notes]
        n_notes = len(pitches)
        if pitches:
            rng = int(max(pitches) - min(pitches))
            med = int(np.median(pitches))
        else:
            rng, med = 0, 0
        per_stem["melody"] = {
            "notes_per_bar": float(n_notes) / n_bars,
            "range_semitones": rng,
            "median_pitch_midi": med,
        }

    return per_stem


def analyze_global(
    aligned_wav: Path,
    *,
    beats: list[float],
    downbeats: list[float],
    bpm: float,
    meter: str,
    duration_seconds: float | None = None,
    extra_warnings: list[str] | None = None,
) -> AnalysisResult:
    """
    Global musical profile from the aligned mix.

    Public stage entrypoint. Beats/downbeats/bpm usually come from timeline stage.
    """
    aligned_wav = Path(aligned_wav)
    y, sr = librosa.load(str(aligned_wav), sr=None, mono=True)
    if duration_seconds is None:
        duration_seconds = float(len(y) / sr)

    key, key_conf, key_warnings = estimate_key(y, sr)
    energy_curve, sections = energy_curve_and_sections(y, sr, downbeats, duration_seconds)

    warnings = list(extra_warnings or [])
    warnings.extend(key_warnings)

    return AnalysisResult(
        bpm=float(bpm),
        meter=meter,
        key=key,
        beats=list(beats),
        downbeats=list(downbeats),
        duration_seconds=float(duration_seconds),
        energy_curve=energy_curve,
        sections=sections,
        warnings=warnings,
        key_confidence=key_conf,
    )


def load_mono(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    data, file_sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr is not None and sr != file_sr:
        data = librosa.resample(data.astype(float), orig_sr=file_sr, target_sr=sr)
        return data.astype(float), sr
    return data.astype(float), int(file_sr)
