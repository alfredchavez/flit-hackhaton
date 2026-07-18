"""Quantize, filter short notes, melody extraction/cleanup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

MIN_NOTE_DURATION = 0.060  # 60 ms


def _beat_grid_sixteenths(beats: list[float], duration: float) -> np.ndarray:
    """Build a 1/16 grid from beat positions (4 sixteenths per beat)."""
    if len(beats) < 2:
        # uniform 120bpm-ish fallback grid
        step = 0.125
        return np.arange(0.0, duration + step, step)

    points = [0.0]
    beat_times = list(beats)
    if beat_times[-1] < duration - 1e-3:
        # extrapolate last interval
        if len(beat_times) >= 2:
            ibi = beat_times[-1] - beat_times[-2]
        else:
            ibi = 0.5
        t = beat_times[-1] + ibi
        while t < duration + ibi:
            beat_times.append(t)
            t += ibi

    for i in range(len(beat_times) - 1):
        b0, b1 = float(beat_times[i]), float(beat_times[i + 1])
        if b1 <= b0:
            continue
        for k in range(4):
            points.append(b0 + (b1 - b0) * (k / 4.0))
    points.append(float(beat_times[-1]))
    grid = np.unique(np.asarray(points, dtype=float))
    grid = grid[(grid >= -1e-6) & (grid <= duration + 1.0)]
    return grid


def _snap(t: float, grid: np.ndarray) -> float:
    if len(grid) == 0:
        return max(0.0, t)
    idx = int(np.argmin(np.abs(grid - t)))
    return float(max(0.0, grid[idx]))


def filter_short_notes(
    pm: pretty_midi.PrettyMIDI,
    min_duration: float = MIN_NOTE_DURATION,
) -> pretty_midi.PrettyMIDI:
    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if (n.end - n.start) >= min_duration]
    return pm


def quantize_midi(
    pm: pretty_midi.PrettyMIDI,
    beats: list[float],
    *,
    duration: float | None = None,
) -> pretty_midi.PrettyMIDI:
    if duration is None:
        duration = pm.get_end_time()
        if duration <= 0:
            duration = beats[-1] if beats else 1.0
    grid = _beat_grid_sixteenths(beats, float(duration))
    for inst in pm.instruments:
        new_notes = []
        for n in inst.notes:
            start = _snap(n.start, grid)
            end = _snap(n.end, grid)
            if end <= start:
                # keep at least one grid step or original length
                if len(grid) >= 2:
                    step = float(np.median(np.diff(grid)))
                else:
                    step = 0.125
                end = start + max(step, MIN_NOTE_DURATION)
            new_notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=start,
                    end=end,
                )
            )
        inst.notes = new_notes
    return pm


def make_monophonic(pm: pretty_midi.PrettyMIDI, *, prefer: str = "highest") -> pretty_midi.PrettyMIDI:
    """Collapse overlapping notes to a single voice."""
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda n: (n.start, -n.pitch if prefer == "highest" else n.pitch))
        kept: list[pretty_midi.Note] = []
        for n in notes:
            if not kept:
                kept.append(n)
                continue
            prev = kept[-1]
            if n.start >= prev.end - 1e-4:
                kept.append(n)
                continue
            # overlap: keep preferred pitch, truncate previous if needed
            if prefer == "highest":
                if n.pitch > prev.pitch:
                    prev.end = max(prev.start + MIN_NOTE_DURATION, n.start)
                    if prev.end <= prev.start:
                        kept.pop()
                    kept.append(n)
                else:
                    # drop n or shorten
                    continue
            else:
                if n.pitch < prev.pitch:
                    prev.end = max(prev.start + MIN_NOTE_DURATION, n.start)
                    if prev.end <= prev.start:
                        kept.pop()
                    kept.append(n)
                else:
                    continue
        # clean zero-length
        inst.notes = [n for n in kept if n.end > n.start]
    return pm


def cleanup_melody_from_vocals(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Dirty vocal → melody cleanup: monophonic upper line, drop extremes."""
    pm = make_monophonic(pm, prefer="highest")
    for inst in pm.instruments:
        inst.is_drum = False
        inst.name = "melody"
        # Drop very low (likely bleed) and ultra-short leftovers
        inst.notes = [
            n
            for n in inst.notes
            if 48 <= n.pitch <= 96 and (n.end - n.start) >= MIN_NOTE_DURATION
        ]
    return pm


def melody_from_other(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Crude upper-register line from other stem."""
    out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    # copy tempo approximately by rebuilding from notes only; tempo set by caller later
    mel = pretty_midi.Instrument(program=0, is_drum=False, name="melody")
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            if n.pitch >= 55:  # upper-ish
                notes.append(n)
    if not notes:
        # take highest notes overall
        all_notes = [n for inst in pm.instruments for n in inst.notes]
        all_notes.sort(key=lambda n: n.pitch, reverse=True)
        notes = all_notes[: max(1, len(all_notes) // 3)]

    tmp = pretty_midi.PrettyMIDI()
    tmp_inst = pretty_midi.Instrument(program=0)
    tmp_inst.notes = [
        pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in notes
    ]
    tmp.instruments.append(tmp_inst)
    tmp = make_monophonic(tmp, prefer="highest")
    if tmp.instruments:
        mel.notes = tmp.instruments[0].notes
    out.instruments.append(mel)
    return out


def _clone_with_tempo(pm: pretty_midi.PrettyMIDI, bpm: float, name: str, is_drum: bool) -> pretty_midi.PrettyMIDI:
    out = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    inst = pretty_midi.Instrument(program=0, is_drum=is_drum, name=name)
    for src in pm.instruments:
        for n in src.notes:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=max(1, min(127, int(n.velocity))),
                    pitch=int(n.pitch),
                    start=float(n.start),
                    end=float(n.end),
                )
            )
    inst.notes.sort(key=lambda n: (n.start, n.pitch))
    out.instruments.append(inst)
    return out


def postprocess_part(
    raw_midi: Path,
    out_midi: Path,
    *,
    part: str,
    bpm: float,
    beats: list[float],
    quantize: bool = True,
    duration: float | None = None,
) -> Path:
    """Filter + optional quantize for one part. Public helper."""
    raw_midi = Path(raw_midi)
    out_midi = Path(out_midi)
    pm = pretty_midi.PrettyMIDI(str(raw_midi))
    pm = filter_short_notes(pm)

    if part in ("bass", "vocals", "melody"):
        prefer = "lowest" if part == "bass" else "highest"
        pm = make_monophonic(pm, prefer=prefer)

    if quantize:
        pm = quantize_midi(pm, beats, duration=duration)

    is_drum = part == "drums"
    pm = _clone_with_tempo(pm, bpm, part, is_drum)
    # Re-apply monophonic/filter already done; drums stay polyphonic across pads.
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi))
    return out_midi


def build_melody(
    vocals_midi: Path,
    other_midi: Path,
    out_midi: Path,
    *,
    bpm: float,
    beats: list[float],
    quantize: bool = True,
    duration: float | None = None,
    min_vocal_notes: int = 8,
) -> tuple[Path, list[str]]:
    """
    Cleaned monophonic melody from vocals, or fallback from other.

    Public stage entrypoint.
    """
    warnings: list[str] = []
    vocals_midi = Path(vocals_midi)
    other_midi = Path(other_midi)
    out_midi = Path(out_midi)

    vpm = pretty_midi.PrettyMIDI(str(vocals_midi))
    v_notes = sum(len(i.notes) for i in vpm.instruments)

    if v_notes >= min_vocal_notes:
        pm = cleanup_melody_from_vocals(vpm)
    else:
        warnings.append("melody_from_other")
        opm = pretty_midi.PrettyMIDI(str(other_midi))
        pm = melody_from_other(opm)

    pm = filter_short_notes(pm)
    pm = make_monophonic(pm, prefer="highest")
    if quantize:
        pm = quantize_midi(pm, beats, duration=duration)
    pm = _clone_with_tempo(pm, bpm, "melody", False)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi))
    return out_midi, warnings
