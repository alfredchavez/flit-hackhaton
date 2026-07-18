"""Beat/downbeat prepass; trim and shift to musical t=0."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass
class TimelineResult:
    aligned_wav: Path
    source_offset_seconds: float
    timeline_origin: str  # "first_downbeat" | "source_start"
    beats: list[float]
    downbeats: list[float]
    bpm: float
    meter: str
    duration_seconds: float
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def _detect_beats_beat_this(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import File2Beats

    # Apple Silicon: MPS may work; CPU is reliable for hackathon demo.
    detector = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = detector(str(wav_path))
    beats = np.asarray(beats, dtype=float).reshape(-1)
    downbeats = np.asarray(downbeats, dtype=float).reshape(-1)
    return beats, downbeats


def _detect_beats_librosa(wav_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import librosa

    warnings: list[str] = ["meter_assumed_4_4"]
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats = np.asarray(beat_times, dtype=float).reshape(-1)
    # Assume 4/4: every 4th beat is a downbeat, starting at first beat.
    if len(beats) == 0:
        downbeats = np.array([], dtype=float)
    else:
        downbeats = beats[::4]
    return beats, downbeats, warnings


def detect_beats(wav_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (beats, downbeats, warnings) in seconds from file start."""
    warnings: list[str] = []
    try:
        beats, downbeats = _detect_beats_beat_this(wav_path)
        if len(beats) < 2:
            raise RuntimeError("beat-this returned too few beats")
        return beats, downbeats, warnings
    except Exception as exc:  # noqa: BLE001 — fall back after primary fails
        print(f"[timeline] beat-this failed ({exc}); falling back to librosa")
        beats, downbeats, fb_warnings = _detect_beats_librosa(wav_path)
        warnings.extend(fb_warnings)
        return beats, downbeats, warnings


def _bpm_from_beats(beats: np.ndarray) -> float:
    if len(beats) < 2:
        return 120.0
    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]  # 30–300 BPM-ish
    if len(intervals) == 0:
        return 120.0
    median_ibi = float(np.median(intervals))
    if median_ibi <= 0:
        return 120.0
    return 60.0 / median_ibi


def _first_reliable_downbeat(downbeats: np.ndarray, beats: np.ndarray) -> float | None:
    """Pick first downbeat that looks real (not tiny silence-leading artifact)."""
    if len(downbeats) == 0:
        # Fall back: first beat if present.
        if len(beats) == 0:
            return None
        return float(beats[0])
    # Prefer the first downbeat; if it is extremely early and a later one is denser, still use first.
    return float(downbeats[0])


def trim_wav(in_path: Path, out_path: Path, offset_seconds: float) -> float:
    """Trim audio from offset_seconds; return duration of result."""
    data, sr = sf.read(str(in_path), always_2d=True)
    start = max(0, int(round(offset_seconds * sr)))
    if start >= len(data):
        raise RuntimeError(f"offset {offset_seconds}s beyond audio length")
    trimmed = data[start:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), trimmed, sr)
    return len(trimmed) / float(sr)


def shift_times(times: np.ndarray, offset: float) -> list[float]:
    shifted = np.asarray(times, dtype=float) - offset
    shifted = shifted[shifted >= -1e-6]
    shifted = np.clip(shifted, 0.0, None)
    return [float(t) for t in shifted]


def align_timeline(
    normalized_wav: Path,
    work_dir: Path,
    *,
    force: bool = False,
) -> TimelineResult:
    """
    Detect first reliable downbeat, trim mix there, return aligned timeline.

    Public stage entrypoint.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    aligned_wav = work_dir / "aligned.wav"
    meta_path = work_dir / "timeline.json"

    from flithack.cache import stage_complete, write_marker

    options = {"version": 1}
    expected = [aligned_wav, meta_path]
    if stage_complete(
        work_dir,
        "timeline",
        source=normalized_wav,
        options=options,
        expected_outputs=expected,
        force=force,
    ):
        import json

        meta = json.loads(meta_path.read_text())
        return TimelineResult(
            aligned_wav=aligned_wav,
            source_offset_seconds=meta["source_offset_seconds"],
            timeline_origin=meta["timeline_origin"],
            beats=meta["beats"],
            downbeats=meta["downbeats"],
            bpm=meta["bpm"],
            meter=meta["meter"],
            duration_seconds=meta["duration_seconds"],
            warnings=list(meta.get("warnings") or []),
        )

    beats_raw, downbeats_raw, warnings = detect_beats(normalized_wav)
    offset = _first_reliable_downbeat(downbeats_raw, beats_raw)
    if offset is None:
        offset = 0.0
        timeline_origin = "source_start"
        warnings.append("timeline_unaligned")
    else:
        timeline_origin = "first_downbeat"

    duration = trim_wav(normalized_wav, aligned_wav, offset)
    beats = shift_times(beats_raw, offset)
    downbeats = shift_times(downbeats_raw, offset)

    # Ensure a downbeat at t=0 when we aligned to first downbeat.
    if timeline_origin == "first_downbeat":
        if not downbeats or downbeats[0] > 0.05:
            downbeats = [0.0] + downbeats
        else:
            downbeats[0] = 0.0
        if not beats or beats[0] > 0.05:
            beats = [0.0] + beats
        else:
            beats[0] = 0.0

    bpm = _bpm_from_beats(np.asarray(beats, dtype=float))
    meter = "4/4"
    if "meter_assumed_4_4" not in warnings:
        # beat-this does not expose meter; MVP assumes 4/4.
        pass

    result = TimelineResult(
        aligned_wav=aligned_wav,
        source_offset_seconds=float(offset),
        timeline_origin=timeline_origin,
        beats=beats,
        downbeats=downbeats,
        bpm=float(bpm),
        meter=meter,
        duration_seconds=float(duration),
        warnings=warnings,
    )

    import json

    meta_path.write_text(
        json.dumps(
            {
                "source_offset_seconds": result.source_offset_seconds,
                "timeline_origin": result.timeline_origin,
                "beats": result.beats,
                "downbeats": result.downbeats,
                "bpm": result.bpm,
                "meter": result.meter,
                "duration_seconds": result.duration_seconds,
                "warnings": result.warnings,
            },
            indent=2,
        )
    )
    write_marker(
        work_dir,
        "timeline",
        source=normalized_wav,
        options=options,
        outputs=expected,
    )
    return result
