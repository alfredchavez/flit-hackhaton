"""Validate and package the analysis_output/ contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pretty_midi

REQUIRED_STEMS = ("drums.wav", "bass.wav", "vocals.wav", "other.wav")
REQUIRED_MIDI = ("drums.mid", "bass.mid", "vocals.mid", "other.mid", "melody.mid")


def build_profile(
    *,
    source_file: str,
    source_offset_seconds: float,
    timeline_origin: str,
    duration_seconds: float,
    bpm: float,
    meter: str,
    key: str,
    beats: list[float],
    downbeats: list[float],
    energy_curve: list[dict[str, float]],
    sections: list[dict[str, Any]],
    chords: list[dict[str, Any]],
    per_stem: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "source_file": source_file,
        "source_offset_seconds": float(source_offset_seconds),
        "timeline_origin": timeline_origin,
        "duration_seconds": float(duration_seconds),
        "bpm": float(bpm),
        "meter": meter,
        "key": key,
        "beats": [float(b) for b in beats],
        "downbeats": [float(d) for d in downbeats],
        "energy_curve": energy_curve,
        "sections": sections,
        "chords": chords,
        "per_stem": per_stem,
        "warnings": list(dict.fromkeys(warnings)),  # stable unique
    }


def validate_output(output_dir: Path) -> None:
    """Raise if required artifacts are missing or unreadable."""
    output_dir = Path(output_dir)
    profile_path = output_dir / "reference_profile.json"
    if not profile_path.is_file():
        raise RuntimeError("missing reference_profile.json")

    profile = json.loads(profile_path.read_text())
    for field in (
        "schema_version",
        "source_file",
        "source_offset_seconds",
        "timeline_origin",
        "duration_seconds",
        "bpm",
        "meter",
        "key",
        "beats",
        "downbeats",
        "energy_curve",
        "chords",
        "per_stem",
        "warnings",
    ):
        if field not in profile:
            raise RuntimeError(f"reference_profile.json missing field: {field}")

    stems = output_dir / "stems"
    midi = output_dir / "midi"
    for name in REQUIRED_STEMS:
        p = stems / name
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"missing or empty stem: {p}")
    for name in REQUIRED_MIDI:
        p = midi / name
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"missing or empty MIDI: {p}")
        # Must open as MIDI
        try:
            pretty_midi.PrettyMIDI(str(p))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"unreadable MIDI {p}: {exc}") from exc


def package_output(
    output_dir: Path,
    *,
    profile: dict[str, Any],
    stem_paths: dict[str, Path],
    midi_paths: dict[str, Path],
) -> Path:
    """
    Assemble analysis_output/ layout and validate.

    Public stage entrypoint.
    """
    output_dir = Path(output_dir)
    stems_dir = output_dir / "stems"
    midi_dir = output_dir / "midi"
    stems_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)

    for name in ("drums", "bass", "vocals", "other"):
        src = Path(stem_paths[name])
        dest = stems_dir / f"{name}.wav"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    for name in ("drums", "bass", "vocals", "other", "melody"):
        src = Path(midi_paths[name])
        dest = midi_dir / f"{name}.mid"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    profile_path = output_dir / "reference_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2))
    validate_output(output_dir)
    return output_dir


def write_fake_fixture(output_dir: Path) -> Path:
    """Hand-written fake analysis_output/ for downstream development."""
    output_dir = Path(output_dir)
    stems_dir = output_dir / "stems"
    midi_dir = output_dir / "midi"
    stems_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf

    sr = 44100
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    for name, freq in (("drums", 80.0), ("bass", 110.0), ("vocals", 440.0), ("other", 330.0)):
        wave = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        sf.write(str(stems_dir / f"{name}.wav"), wave, sr)

    bpm = 120.0
    for name, is_drum, pitches in (
        ("drums", True, [36, 38, 42]),
        ("bass", False, [36, 38, 41, 43]),
        ("vocals", False, [60, 62, 64, 65]),
        ("other", False, [48, 52, 55, 60]),
        ("melody", False, [72, 74, 76, 77]),
    ):
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        inst = pretty_midi.Instrument(program=0, is_drum=is_drum, name=name)
        for i, p in enumerate(pitches):
            start = i * 0.5
            inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=start, end=start + 0.4))
        pm.instruments.append(inst)
        pm.write(str(midi_dir / f"{name}.mid"))

    profile = build_profile(
        source_file="fake_reference.mp3",
        source_offset_seconds=0.0,
        timeline_origin="first_downbeat",
        duration_seconds=2.0,
        bpm=bpm,
        meter="4/4",
        key="C major",
        beats=[0.0, 0.5, 1.0, 1.5],
        downbeats=[0.0, 2.0],
        energy_curve=[
            {"start": 0.0, "end": 2.0, "value": 0.5},
        ],
        sections=[{"start": 0.0, "end": 2.0, "energy": "medium"}],
        chords=[
            {"start": 0.0, "end": 1.0, "chord": "C"},
            {"start": 1.0, "end": 2.0, "chord": "G"},
        ],
        per_stem={
            "drums": {"onsets_per_bar": 3.0},
            "bass": {
                "notes_per_bar": 4.0,
                "register_low_midi": 36,
                "register_high_midi": 43,
            },
            "melody": {
                "notes_per_bar": 4.0,
                "range_semitones": 5,
                "median_pitch_midi": 74,
            },
        },
        warnings=[],
    )
    (output_dir / "reference_profile.json").write_text(json.dumps(profile, indent=2))
    validate_output(output_dir)
    return output_dir
