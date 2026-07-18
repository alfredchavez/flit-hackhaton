"""MIDI → compact text so an LLM can read the transcription."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pretty_midi

DRUM_MAP = {
    36: "KICK",
    38: "SNARE",
    42: "CLOSED_HAT",
    47: "TOM",
    49: "CYMBAL",
}

# Short labels for grid rows
_DRUM_ROW = {
    "KICK": "KICK",
    "SNARE": "SNARE",
    "CLOSED_HAT": "HAT",
    "TOM": "TOM",
    "CYMBAL": "CYMBAL",
}

_PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _load_profile(analysis_output_dir: Path) -> dict[str, Any]:
    path = analysis_output_dir / "reference_profile.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text())


def _bar_edges(profile: dict[str, Any]) -> list[tuple[float, float]]:
    """Return [(start_sec, end_sec), ...] for each bar from downbeats/energy_curve."""
    duration = float(profile.get("duration_seconds") or 0.0)
    energy = profile.get("energy_curve") or []
    if energy:
        return [(float(e["start"]), float(e["end"])) for e in energy]

    downbeats = list(profile.get("downbeats") or [])
    if not downbeats:
        return [(0.0, duration or 1.0)]
    if downbeats[0] > 0.01:
        downbeats = [0.0] + downbeats
    if downbeats[-1] < duration - 0.05:
        downbeats = downbeats + [duration]
    bars = []
    for i in range(len(downbeats) - 1):
        s, e = float(downbeats[i]), float(downbeats[i + 1])
        if e > s:
            bars.append((s, e))
    return bars or [(0.0, duration or 1.0)]


def select_bar_indices(profile: dict[str, Any], n_first: int = 8, n_peak: int = 8) -> list[int]:
    """
    First 8 bars + consecutive 8-bar window with highest summed energy.
    Deduplicated, sorted, handles short songs.
    """
    bars = _bar_edges(profile)
    n = len(bars)
    if n == 0:
        return []
    energy = profile.get("energy_curve") or []
    values = [float(e.get("value", 0.0)) for e in energy] if energy else [0.0] * n
    if len(values) < n:
        values = values + [0.0] * (n - len(values))
    values = values[:n]

    selected: set[int] = set(range(min(n_first, n)))

    window = min(n_peak, n)
    if window > 0:
        best_start = 0
        best_sum = -1.0
        for i in range(0, n - window + 1):
            s = sum(values[i : i + window])
            if s > best_sum:
                best_sum = s
                best_start = i
        for i in range(best_start, best_start + window):
            selected.add(i)

    return sorted(selected)


def _sec_to_beat(t: float, beats: list[float]) -> float:
    """Map absolute seconds to beat position (1-based within track: beat 1 = first beat)."""
    if not beats:
        return 0.0
    if t <= beats[0]:
        return 1.0
    if t >= beats[-1]:
        # extrapolate last interval
        if len(beats) >= 2:
            ibi = beats[-1] - beats[-2]
        else:
            ibi = 0.5
        if ibi <= 0:
            ibi = 0.5
        return float(len(beats)) + (t - beats[-1]) / ibi

    # binary search segment
    lo, hi = 0, len(beats) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if beats[mid] <= t:
            lo = mid
        else:
            hi = mid
    b0, b1 = beats[lo], beats[hi]
    if b1 <= b0:
        return float(lo + 1)
    frac = (t - b0) / (b1 - b0)
    return float(lo + 1) + frac


def _note_name(pitch: int) -> str:
    try:
        return pretty_midi.note_number_to_name(int(pitch))
    except Exception:
        pc = _PC_NAMES[int(pitch) % 12]
        octave = int(pitch) // 12 - 1
        return f"{pc}{octave}"


def _vel_bucket(v: int) -> str:
    if v >= 100:
        return "accent"
    if v >= 70:
        return "med"
    return "soft"


def _vel_char(v: int) -> str:
    """Drum grid: x = accent, o = soft/normal."""
    return "x" if v >= 90 else "o"


def _load_notes(midi_path: Path) -> list[pretty_midi.Note]:
    if not midi_path.is_file():
        return []
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    return notes


def _compress_chords(chords: list[dict[str, Any]], bar_start: float, bar_end: float) -> str:
    """Run-length-compress chords overlapping [bar_start, bar_end)."""
    hits = []
    for c in chords:
        cs, ce = float(c["start"]), float(c["end"])
        if ce <= bar_start or cs >= bar_end:
            continue
        hits.append(str(c.get("chord", "N")))
    if not hits:
        return "—"
    compressed: list[str] = []
    i = 0
    while i < len(hits):
        j = i + 1
        while j < len(hits) and hits[j] == hits[i]:
            j += 1
        n = j - i
        if n > 1:
            compressed.append(f"{hits[i]}×{n}")
        else:
            compressed.append(hits[i])
        i = j
    return " → ".join(compressed)


def _header(profile: dict[str, Any], selected: list[int], non_4_4: bool) -> str:
    lines = [
        "=== REFERENCE TRANSCRIPTION (compact) ===",
        f"source: {profile.get('source_file', '?')}",
        f"bpm: {profile.get('bpm')}  key: {profile.get('key')}  meter: {profile.get('meter')}  "
        f"duration_s: {profile.get('duration_seconds')}",
        f"timeline_origin: {profile.get('timeline_origin')}  offset_s: {profile.get('source_offset_seconds')}",
    ]
    if non_4_4:
        lines.append("flags: midi_repr_non_4_4")

    sections = profile.get("sections") or []
    if sections:
        sec_bits = [
            f"{s.get('energy', '?')}[{s.get('start', 0):.1f}-{s.get('end', 0):.1f}s]" for s in sections
        ]
        lines.append("sections: " + ", ".join(sec_bits))

    # Compressed chord summary (unique sequence, not full array)
    chords = profile.get("chords") or []
    if chords:
        seq: list[str] = []
        for c in chords:
            name = str(c.get("chord", "N"))
            if not seq or seq[-1] != name:
                seq.append(name)
        # cap length
        if len(seq) > 24:
            shown = " → ".join(seq[:12]) + " … " + " → ".join(seq[-8:])
        else:
            shown = " → ".join(seq)
        lines.append(f"chord_summary ({len(seq)} changes): {shown}")
    else:
        lines.append("chord_summary: (none / unreliable)")

    energy = profile.get("energy_curve") or []
    if energy:
        vals = [float(e.get("value", 0)) for e in energy]
        avg = sum(vals) / len(vals)
        peak = max(vals)
        low = min(vals)
        lines.append(
            f"energy: bars={len(vals)} avg={avg:.2f} peak={peak:.2f} low={low:.2f}"
        )

    per = profile.get("per_stem") or {}
    if per:
        bits = []
        if "drums" in per:
            bits.append(f"drums_onsets/bar={per['drums'].get('onsets_per_bar', '?')}")
        if "bass" in per:
            b = per["bass"]
            bits.append(
                f"bass_notes/bar={b.get('notes_per_bar', '?')} "
                f"reg={b.get('register_low_midi', '?')}-{b.get('register_high_midi', '?')}"
            )
        if "melody" in per:
            m = per["melody"]
            bits.append(
                f"melody_notes/bar={m.get('notes_per_bar', '?')} "
                f"range={m.get('range_semitones', '?')}st med={m.get('median_pitch_midi', '?')}"
            )
        if bits:
            lines.append("densities: " + " | ".join(bits))

    warnings = profile.get("warnings") or []
    if warnings:
        lines.append("warnings: " + ", ".join(warnings))

    lines.append(f"selected_bars (0-based): {selected}  (first 8 + peak-energy window)")
    lines.append("")
    return "\n".join(lines)


def _serialize_drums_grid(
    notes: list[pretty_midi.Note],
    bars: list[tuple[float, float]],
    selected: list[int],
) -> str:
    lines = ["--- DRUMS (step grids: x=accent o=soft .=rest; 16 steps/bar) ---"]
    for bi in selected:
        if bi < 0 or bi >= len(bars):
            continue
        start, end = bars[bi]
        dur = end - start
        if dur <= 0:
            continue
        # collect hits per instrument: list of (step, velocity)
        by_inst: dict[str, list[tuple[int, int]]] = {k: [] for k in _DRUM_ROW}
        for n in notes:
            # Drum timing is defined by the onset. Using a note's midpoint shifts
            # quantized hits later because drum notes still have a short duration.
            if not (start <= n.start < end):
                continue
            label = DRUM_MAP.get(int(n.pitch))
            if label is None:
                # closest known or skip
                continue
            step = int(round((n.start - start) / dur * 16))
            step = max(0, min(15, step))
            by_inst[label].append((step, int(n.velocity)))

        bar_num = bi + 1
        first = True
        for label, short in _DRUM_ROW.items():
            hits = by_inst[label]
            grid = ["."] * 16
            for step, vel in hits:
                ch = _vel_char(vel)
                # accent wins over soft if both land
                if grid[step] == "." or (ch == "x" and grid[step] == "o"):
                    grid[step] = ch
            row = "".join(grid)
            if first:
                lines.append(f"bar {bar_num} | {short:6s} {row}")
                first = False
            else:
                lines.append(f"      | {short:6s} {row}")
        if first:
            lines.append(f"bar {bar_num} | (empty)")
    return "\n".join(lines)


def _serialize_drums_events(
    notes: list[pretty_midi.Note],
    bars: list[tuple[float, float]],
    selected: list[int],
    beats: list[float],
) -> str:
    lines = ["--- DRUMS (beat-position events; non-4/4) ---"]
    for bi in selected:
        if bi < 0 or bi >= len(bars):
            continue
        start, end = bars[bi]
        bar_beat0 = _sec_to_beat(start, beats)
        events = []
        for n in notes:
            if not (start <= n.start < end):
                continue
            label = DRUM_MAP.get(int(n.pitch), f"P{n.pitch}")
            pos = _sec_to_beat(n.start, beats) - bar_beat0 + 1.0
            events.append(f"{label}@{pos:.2f}v{_vel_bucket(n.velocity)}")
        lines.append(f"bar {bi + 1}: " + (" | ".join(events) if events else "(empty)"))
    return "\n".join(lines)


def _serialize_pitched(
    name: str,
    notes: list[pretty_midi.Note],
    bars: list[tuple[float, float]],
    selected: list[int],
    beats: list[float],
) -> str:
    lines = [f"--- {name.upper()} (events: Note@beat lenBeats [vel]) ---"]
    for bi in selected:
        if bi < 0 or bi >= len(bars):
            continue
        start, end = bars[bi]
        bar_beat0 = _sec_to_beat(start, beats)
        events = []
        for n in sorted(notes, key=lambda x: x.start):
            if not (start <= n.start < end):
                continue
            pos = _sec_to_beat(n.start, beats) - bar_beat0 + 1.0
            end_b = _sec_to_beat(n.end, beats) - bar_beat0 + 1.0
            length = max(0.05, end_b - pos)
            events.append(
                f"{_note_name(n.pitch)}@{pos:.1f} len{length:.1f} {_vel_bucket(n.velocity)}"
            )
        lines.append(
            f"{name} bar {bi + 1}: " + (" | ".join(events) if events else "(empty)")
        )
    return "\n".join(lines)


def _serialize_other(
    notes: list[pretty_midi.Note],
    bars: list[tuple[float, float]],
    selected: list[int],
) -> str:
    lines = ["--- OTHER (pitch-class set + register span; messy harmonic sketch) ---"]
    for bi in selected:
        if bi < 0 or bi >= len(bars):
            continue
        start, end = bars[bi]
        # Include held notes that began in the previous bar but still sound here.
        in_bar = [n for n in notes if n.start < end and n.end > start]
        if not in_bar:
            lines.append(f"other bar {bi + 1}: (empty)")
            continue
        pcs = sorted({_PC_NAMES[n.pitch % 12] for n in in_bar}, key=lambda x: _PC_NAMES.index(x) if x in _PC_NAMES else 99)
        lo = min(n.pitch for n in in_bar)
        hi = max(n.pitch for n in in_bar)
        lines.append(
            f"other bar {bi + 1}: {{{','.join(pcs)}}} span {_note_name(lo)}-{_note_name(hi)}, {len(in_bar)} notes"
        )
    return "\n".join(lines)


def _selected_chords_block(
    profile: dict[str, Any],
    bars: list[tuple[float, float]],
    selected: list[int],
) -> str:
    chords = profile.get("chords") or []
    lines = ["--- CHORDS (selected bars only, RLE) ---"]
    if not chords:
        lines.append("(no chords / chords_unreliable)")
        return "\n".join(lines)
    for bi in selected:
        if bi < 0 or bi >= len(bars):
            continue
        start, end = bars[bi]
        lines.append(f"bar {bi + 1}: {_compress_chords(chords, start, end)}")
    return "\n".join(lines)


def midi_repr(analysis_output_dir: str | Path) -> str:
    """
    Serialize analysis_output/ MIDI + profile into compact LLM-readable text.

    Public stage entrypoint.
    """
    root = Path(analysis_output_dir)
    profile = _load_profile(root)
    bars = _bar_edges(profile)
    selected = select_bar_indices(profile)
    beats = [float(b) for b in (profile.get("beats") or [])]
    meter = str(profile.get("meter") or "4/4")
    is_4_4 = meter.replace(" ", "") == "4/4"

    midi_dir = root / "midi"
    drums = _load_notes(midi_dir / "drums.mid")
    bass = _load_notes(midi_dir / "bass.mid")
    melody = _load_notes(midi_dir / "melody.mid")
    other = _load_notes(midi_dir / "other.mid")

    parts = [
        _header(profile, selected, non_4_4=not is_4_4),
        _selected_chords_block(profile, bars, selected),
        "",
    ]
    if is_4_4:
        parts.append(_serialize_drums_grid(drums, bars, selected))
    else:
        parts.append(_serialize_drums_events(drums, bars, selected, beats))
    parts.extend(
        [
            "",
            _serialize_pitched("bass", bass, bars, selected, beats),
            "",
            _serialize_pitched("melody", melody, bars, selected, beats),
            "",
            _serialize_other(other, bars, selected),
        ]
    )
    text = "\n".join(parts).rstrip() + "\n"
    # crude token count for development
    print(f"[midi_repr] ~{len(text) // 4} tokens (len={len(text)})")
    return text
