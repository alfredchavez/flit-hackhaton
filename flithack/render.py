"""Deterministic plan + parts → valid MIDI files. No network."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mido
import pretty_midi

# GM drums (spec)
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_CLOSED_HAT = 42
DRUM_TOM = 47
DRUM_CYMBAL = 49

DRUM_ROWS = {
    "kick": DRUM_KICK,
    "snare": DRUM_SNARE,
    "closed_hat": DRUM_CLOSED_HAT,
    "tom": DRUM_TOM,
    "cymbal": DRUM_CYMBAL,
}

VEL = {"accent": 100, "med": 80, "soft": 60, "x": 100, "o": 60}

# Fixed preview programs (zero-based GM)
PROG_BASS = 33  # Electric Bass (finger)
PROG_HARMONY = 48  # String Ensemble 1
PROG_MELODY = 80  # Lead 1 (square)

_NOTE_RE = re.compile(
    r"(?P<note>[A-Ga-g](?:#|b)?)(?P<oct>-?\d+)"
    r"@(?P<beat>\d+(?:\.\d+)?)"
    r"\s+len(?P<dur>\d+(?:\.\d+)?)"
    r"(?:\s+(?P<vel>accent|med|soft))?",
    re.IGNORECASE,
)

_PC = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def _snap_16(beat: float) -> float:
    """Snap beat position to 1/16 grid (0.25 beat steps). Beat is absolute from song start."""
    return round(beat * 4.0) / 4.0


def _sec(beat: float, bpm: float) -> float:
    return (beat * 60.0) / float(bpm)


def note_name_to_midi(name: str) -> int | None:
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", name.strip())
    if not m:
        return None
    letter, acc, oct_s = m.group(1).upper(), m.group(2), m.group(3)
    key = letter + ({"#": "#", "b": "B"}.get(acc, "") if acc == "#" else ("B" if acc == "b" else ""))
    # normalize flats
    if acc == "b":
        key = letter + "B"
    elif acc == "#":
        key = letter + "#"
    else:
        key = letter
    # map
    table = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }
    if key not in table and len(key) == 2 and key[1] == "B":
        # already Db style as DB
        pass
    pc = table.get(key)
    if pc is None:
        return None
    octave = int(oct_s)
    return 12 * (octave + 1) + pc


def parse_chord_symbol(sym: str) -> tuple[int, list[int], str] | None:
    """
    Return (root_pc, intervals, quality_tag) or None if unusable.
    intervals are semitones from root for the voicing.
    """
    s = (sym or "").strip()
    if not s or s.upper() in ("N", "NC", "NONE", "-"):
        return None
    m = re.match(r"^([A-Ga-g])([#b]?)(.*)$", s)
    if not m:
        return None
    letter, acc, rest = m.group(1).upper(), m.group(2), m.group(3).strip()
    root_key = letter + (acc if acc else "")
    # normalize flat notation for table
    if acc == "b":
        root_key = letter + "B"
    root_table = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }
    if root_key not in root_table and acc == "b":
        root_key = letter + "B"
    root = root_table.get(root_key)
    if root is None:
        return None

    raw_quality = rest.replace(" ", "")
    r = raw_quality.lower()
    quality = "maj"
    intervals = [0, 4, 7]
    known = True
    if raw_quality in ("M", "MAJ", "Major") or r in ("", "maj", "major"):
        intervals = [0, 4, 7]
        quality = "maj"
    elif r in ("m", "min", "minor", "-"):
        intervals = [0, 3, 7]
        quality = "min"
    elif r in ("7", "dom7"):
        intervals = [0, 4, 7, 10]
        quality = "7"
    elif raw_quality in ("M7", "MAJ7") or r in ("maj7", "Δ", "Δ7"):
        intervals = [0, 4, 7, 11]
        quality = "maj7"
    elif r in ("m7", "min7", "-7"):
        intervals = [0, 3, 7, 10]
        quality = "m7"
    elif r in ("sus", "sus4"):
        intervals = [0, 5, 7]
        quality = "sus4"
    elif r in ("sus2",):
        intervals = [0, 2, 7]
        quality = "sus2"
    elif r in ("dim", "o"):
        intervals = [0, 3, 6]
        quality = "dim"
    elif r in ("aug", "+"):
        intervals = [0, 4, 8]
        quality = "aug"
    else:
        # unknown quality but valid root → plain triad + caller warns
        intervals = [0, 4, 7]
        quality = "maj"
        known = False
        return root, intervals, f"unknown:{rest}" if rest else "maj"

    return root, intervals, quality if known else f"unknown:{rest}"


def key_tonic_triad(key: str) -> str:
    """Return a simple chord symbol for the key's tonic triad."""
    k = (key or "C major").strip()
    m = re.match(r"^([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m)?", k, re.I)
    if not m:
        return "C"
    root = m.group(1).upper() + (m.group(2) or "")
    mode = (m.group(3) or "major").lower()
    if mode in ("minor", "min", "m"):
        return root + "m"
    return root


def _repair_grid(row: str, warnings: list[str], label: str) -> str:
    chars = []
    for ch in (row or ""):
        if ch in "xo.":
            chars.append(ch)
        elif ch in "XO*":
            chars.append("x" if ch != "O" else "o")
        # ignore others
    if len(chars) < 16:
        if row:
            warnings.append(f"drum_grid_padded:{label}:{len(chars)}")
        chars = chars + ["."] * (16 - len(chars))
    elif len(chars) > 16:
        warnings.append(f"drum_grid_truncated:{label}:{len(chars)}")
        chars = chars[:16]
    return "".join(chars)


def _section_bar_offsets(plan: dict[str, Any]) -> list[tuple[dict, int, int]]:
    """List of (section, start_bar_index, n_bars) absolute 0-based bar indices."""
    out = []
    bar = 0
    for sec in plan.get("sections") or []:
        n = int(sec.get("bars") or 0)
        out.append((sec, bar, n))
        bar += n
    return out


def _empty_pm(bpm: float) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(initial_tempo=float(bpm))


def _add_note(
    inst: pretty_midi.Instrument,
    pitch: int,
    start_beat: float,
    end_beat: float,
    bpm: float,
    velocity: int,
) -> None:
    if end_beat <= start_beat:
        return
    start_beat = _snap_16(start_beat)
    end_beat = _snap_16(end_beat)
    if end_beat <= start_beat:
        end_beat = start_beat + 0.25
    inst.notes.append(
        pretty_midi.Note(
            velocity=max(1, min(127, int(velocity))),
            pitch=int(pitch),
            start=_sec(start_beat, bpm),
            end=_sec(end_beat, bpm),
        )
    )


def _dedupe_same_pitch(inst: pretty_midi.Instrument) -> None:
    """No overlapping same-pitch notes."""
    by_pitch: dict[int, list[pretty_midi.Note]] = {}
    for n in sorted(inst.notes, key=lambda x: (x.pitch, x.start)):
        by_pitch.setdefault(n.pitch, []).append(n)
    kept: list[pretty_midi.Note] = []
    for pitch, notes in by_pitch.items():
        cur: pretty_midi.Note | None = None
        for n in notes:
            if cur is None:
                cur = n
                continue
            if n.start < cur.end - 1e-6:
                # overlap: extend or keep louder
                if n.velocity >= cur.velocity:
                    cur.end = max(cur.end, n.end)
                else:
                    cur.end = max(cur.end, n.end)
            else:
                kept.append(cur)
                cur = n
        if cur is not None:
            kept.append(cur)
    inst.notes = sorted(kept, key=lambda x: (x.start, x.pitch))


def _clamp_range(pitch: int, lo: int, hi: int) -> tuple[int, bool]:
    """Octave-shift into [lo, hi]. Returns (pitch, shifted)."""
    if lo <= pitch <= hi:
        return pitch, False
    p = pitch
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    # if still out (range smaller than octave span impossible), clip
    p = max(lo, min(hi, p))
    return p, p != pitch


def parse_event_string(s: str, warnings: list[str], ctx: str) -> list[tuple[int, float, float, int]]:
    """
    Parse one bar event string → list of (pitch, beat_in_bar 1-5, duration_beats, velocity).
    """
    if not s or not str(s).strip():
        return []
    events = []
    for token in str(s).split("|"):
        token = token.strip()
        if not token:
            continue
        m = _NOTE_RE.search(token)
        if not m:
            warnings.append(f"bad_event_token:{ctx}:{token[:40]}")
            continue
        pitch = note_name_to_midi(m.group("note") + m.group("oct"))
        if pitch is None:
            warnings.append(f"bad_pitch:{ctx}:{m.group('note')}{m.group('oct')}")
            continue
        beat = float(m.group("beat"))
        dur = float(m.group("dur"))
        vel_s = (m.group("vel") or "med").lower()
        vel = VEL.get(vel_s, 80)
        if not (1.0 <= beat < 5.0):
            # repair: clamp into bar
            beat = min(4.75, max(1.0, beat))
            warnings.append(f"beat_clamped:{ctx}")
        if dur <= 0:
            warnings.append(f"bad_duration:{ctx}")
            continue
        # convert beat 1.0-based → 0-based within bar
        beat0 = beat - 1.0
        events.append((pitch, beat0, dur, vel))
    return events


def _render_drums(
    plan: dict[str, Any],
    parts: dict[str, Any],
    bpm: float,
    song_bars: int,
    warnings: list[str],
) -> pretty_midi.PrettyMIDI:
    pm = _empty_pm(bpm)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    patterns = {p.get("section_id"): p for p in (parts.get("drums") or [])}

    for sec, start_bar, n_bars in _section_bar_offsets(plan):
        active = sec.get("active_parts") or []
        if "drums" not in active:
            continue
        pat = patterns.get(sec.get("id"))
        if not pat:
            warnings.append(f"part_dropped:drums:{sec.get('id')}")
            continue
        grids = {}
        for row_name in DRUM_ROWS:
            grids[row_name] = _repair_grid(str(pat.get(row_name) or ""), warnings, row_name)
        fill = bool(pat.get("fill_last_bar"))

        for bi in range(n_bars):
            abs_bar = start_bar + bi
            is_last = bi == n_bars - 1
            for row_name, pitch in DRUM_ROWS.items():
                grid = grids[row_name]
                for step, ch in enumerate(grid):
                    if is_last and fill and step >= 8:
                        continue  # fill replaces second half
                    if ch not in ("x", "o"):
                        continue
                    vel = 100 if ch == "x" else 60
                    beat = abs_bar * 4.0 + step * 0.25
                    _add_note(inst, pitch, beat, beat + 0.2, bpm, vel)
            if is_last and fill:
                # snare/tom 16th run on last half-bar
                for step in range(8, 16):
                    pitch = DRUM_SNARE if step % 2 == 0 else DRUM_TOM
                    beat = abs_bar * 4.0 + step * 0.25
                    _add_note(inst, pitch, beat, beat + 0.15, bpm, 100 if step % 2 == 0 else 80)

    _dedupe_same_pitch(inst)
    # ensure end of track at song length
    end_beat = song_bars * 4.0
    if not inst.notes:
        # keep empty but valid
        pass
    else:
        # clip past end
        for n in inst.notes:
            if n.end > _sec(end_beat, bpm):
                n.end = _sec(end_beat, bpm)
        inst.notes = [n for n in inst.notes if n.end > n.start]
    pm.instruments.append(inst)
    return pm


def _voice_lead(
    prev: list[int] | None,
    root: int,
    intervals: list[int],
) -> list[int]:
    """Nearest-inversion voicing around C3–C5 (48–72)."""
    # base chord tones in mid range
    candidates = []
    for iv in intervals[:4]:
        # pick octave so pitch near 60
        p = root + iv
        while p < 48:
            p += 12
        while p > 72:
            p -= 12
        candidates.append(p)
    candidates = sorted(set(candidates))[:4]
    if not prev:
        return sorted(candidates)

    voiced = []
    used = set()
    for prev_p in prev:
        # keep common tones
        best = None
        best_d = 999
        for c in candidates:
            if c in used:
                continue
            # try octave variants
            for shift in (0, 12, -12, 24, -24):
                p = c + shift
                if p < 48 or p > 84:
                    continue
                d = abs(p - prev_p)
                if d < best_d:
                    best_d = d
                    best = p
        if best is not None:
            voiced.append(best)
            # mark pc used
            used.add(((best - root) % 12) + root)  # rough
            # also remove matching candidate pcs
            candidates = [c for c in candidates if (c % 12) != (best % 12)]
    for c in candidates:
        p = c
        while p < 48:
            p += 12
        while p > 72:
            p -= 12
        if p not in voiced:
            voiced.append(p)
    return sorted(voiced)[:4] or [60, 64, 67]


def _render_harmony(
    plan: dict[str, Any],
    bpm: float,
    song_bars: int,
    warnings: list[str],
) -> pretty_midi.PrettyMIDI:
    pm = _empty_pm(bpm)
    inst = pretty_midi.Instrument(program=PROG_HARMONY, is_drum=False, name="harmony")
    key = plan.get("key") or "C major"
    default_chord = key_tonic_triad(key)
    prev_voicing: list[int] | None = None
    last_valid: tuple[int, list[int]] | None = None

    for sec, start_bar, n_bars in _section_bar_offsets(plan):
        active = sec.get("active_parts") or []
        if "harmony" not in active:
            continue
        chords = list(sec.get("chords") or [])
        while len(chords) < n_bars:
            chords.append(chords[-1] if chords else default_chord)
        chords = chords[:n_bars]

        for bi, sym in enumerate(chords):
            parsed = parse_chord_symbol(str(sym))
            if parsed is None:
                if last_valid:
                    root, intervals = last_valid
                    warnings.append(f"chord_fallback_prev:{sym}")
                else:
                    p2 = parse_chord_symbol(default_chord) or (0, [0, 4, 7], "maj")
                    root, intervals = p2[0], p2[1]
                    warnings.append(f"chord_fallback_tonic:{sym}")
            else:
                root, intervals, quality = parsed
                if quality.startswith("unknown:"):
                    warnings.append(f"chord_unknown_quality:{sym}")
                last_valid = (root, intervals)

            voicing = _voice_lead(prev_voicing, root, intervals)
            prev_voicing = voicing
            abs_bar = start_bar + bi
            start_b = abs_bar * 4.0
            end_b = start_b + 4.0
            for pitch in voicing:
                _add_note(inst, pitch, start_b, end_b - 0.05, bpm, 70)

    _dedupe_same_pitch(inst)
    pm.instruments.append(inst)
    return pm


def _render_phrase_part(
    name: str,
    plan: dict[str, Any],
    phrases: list[dict[str, Any]],
    bpm: float,
    song_bars: int,
    warnings: list[str],
    *,
    program: int,
    pitch_lo: int,
    pitch_hi: int,
    max_phrases_per_section: int = 1,
) -> pretty_midi.PrettyMIDI:
    pm = _empty_pm(bpm)
    inst = pretty_midi.Instrument(program=program, is_drum=False, name=name)

    # group phrases by section_id
    by_sec: dict[str, list[dict]] = {}
    for ph in phrases or []:
        sid = ph.get("section_id")
        if not sid:
            continue
        by_sec.setdefault(sid, []).append(ph)

    for sec, start_bar, n_bars in _section_bar_offsets(plan):
        active = sec.get("active_parts") or []
        if name not in active:
            continue
        plist = by_sec.get(sec.get("id"), [])
        if not plist:
            warnings.append(f"part_dropped:{name}:{sec.get('id')}")
            continue
        if len(plist) > max_phrases_per_section:
            warnings.append(f"extra_{name}_phrases_dropped:{sec.get('id')}")
            plist = plist[:max_phrases_per_section]

        notes_before_section = len(inst.notes)

        # for melody with 2 phrases: alternate complete phrase cycles
        # build list of bar-event lists per phrase
        phrase_bars: list[list[str]] = []
        for ph in plist:
            ebb = list(ph.get("events_by_bar") or [])
            if len(ebb) < 1:
                ebb = [""]
            if len(ebb) > 4:
                warnings.append(f"phrase_truncated:{name}:{sec.get('id')}")
                ebb = ebb[:4]
            phrase_bars.append([str(x) for x in ebb])

        # walk bars
        bar_in_phrase = 0
        current_phrase = 0
        for bi in range(n_bars):
            if not phrase_bars:
                break
            ebb = phrase_bars[current_phrase]
            event_str = ebb[bar_in_phrase % len(ebb)]
            abs_bar = start_bar + bi
            events = parse_event_string(event_str, warnings, f"{name}:{sec.get('id')}:b{bi}")
            for pitch, beat0, dur, vel in events:
                pitch2, shifted = _clamp_range(pitch, pitch_lo, pitch_hi)
                if shifted:
                    warnings.append(f"pitch_octave_shift:{name}:{pitch}->{pitch2}")
                start_b = abs_bar * 4.0 + beat0
                end_b = start_b + dur
                # Notes may sustain across bars; clip only to the section boundary.
                sec_end = (start_bar + n_bars) * 4.0
                end_b = min(end_b, sec_end)
                if end_b > start_b:
                    _add_note(inst, pitch2, start_b, end_b, bpm, vel)

            bar_in_phrase += 1
            if bar_in_phrase >= len(ebb):
                bar_in_phrase = 0
                if len(phrase_bars) > 1:
                    current_phrase = (current_phrase + 1) % len(phrase_bars)

        if len(inst.notes) == notes_before_section:
            warnings.append(f"part_dropped:{name}:{sec.get('id')}")

    _dedupe_same_pitch(inst)
    # clip to song
    end_sec = _sec(song_bars * 4.0, bpm)
    for n in inst.notes:
        if n.end > end_sec:
            n.end = end_sec
    inst.notes = [n for n in inst.notes if n.end > n.start]
    pm.instruments.append(inst)
    return pm


def _track_name(track: mido.MidiTrack) -> str | None:
    for msg in track:
        if msg.is_meta and msg.type == "track_name":
            return str(msg.name)
    return None


def _set_meta_track_end(track: mido.MidiTrack, target_tick: int) -> None:
    """Put a non-audible marker and end-of-track at an exact absolute tick."""
    absolute = 0
    kept: list[tuple[int, mido.Message | mido.MetaMessage]] = []
    for msg in track:
        absolute += int(msg.time)
        if msg.type == "end_of_track":
            continue
        if msg.is_meta and msg.type == "text" and msg.text == "flithack_song_end":
            continue
        kept.append((absolute, msg))

    last_tick = max((tick for tick, _ in kept), default=0)
    target_tick = max(target_tick, last_tick)
    track.clear()
    previous = 0
    for tick, msg in kept:
        track.append(msg.copy(time=tick - previous))
        previous = tick
    track.append(
        mido.MetaMessage(
            "text",
            text="flithack_song_end",
            time=target_tick - previous,
        )
    )
    track.append(mido.MetaMessage("end_of_track", time=0))


def _finalize_midi_file(
    path: Path,
    song_bars: int,
    *,
    expected_tracks: list[tuple[str, int, bool]],
) -> None:
    """Enforce exact song length and retain named tracks even when they are empty."""
    midi = mido.MidiFile(path)
    if not midi.tracks:
        midi.tracks.append(mido.MidiTrack())
    target_tick = int(round(song_bars * 4.0 * midi.ticks_per_beat))
    _set_meta_track_end(midi.tracks[0], target_tick)

    existing = {_track_name(track) for track in midi.tracks[1:]}
    for name, program, is_drum in expected_tracks:
        if name in existing:
            continue
        channel = 9 if is_drum else 0
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        if not is_drum:
            track.append(
                mido.Message(
                    "program_change",
                    program=program,
                    channel=channel,
                    time=0,
                )
            )
        track.append(mido.MetaMessage("end_of_track", time=target_tick))
        midi.tracks.append(track)
    midi.save(path)


def render(plan: dict[str, Any], parts: dict[str, Any], out_dir: str | Path) -> list[str]:
    """
    Deterministic render. Writes midi/drums|bass|harmony|melody.mid and song.mid.

    Public stage entrypoint. Returns warnings.
    """
    out_dir = Path(out_dir)
    midi_dir = out_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = list(plan.get("warnings") or []) + list(parts.get("warnings") or [])

    bpm = float(plan.get("bpm") or 120.0)
    song_bars = sum(int(s.get("bars") or 0) for s in (plan.get("sections") or []))
    if song_bars <= 0:
        song_bars = 16
        warnings.append("render_default_16_bars")

    try:
        drums_pm = _render_drums(plan, parts, bpm, song_bars, warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"part_dropped:drums:render_error:{exc}")
        drums_pm = _empty_pm(bpm)
        drums_pm.instruments.append(pretty_midi.Instrument(program=0, is_drum=True, name="drums"))

    try:
        bass_pm = _render_phrase_part(
            "bass",
            plan,
            parts.get("bass") or [],
            bpm,
            song_bars,
            warnings,
            program=PROG_BASS,
            pitch_lo=28,  # E1
            pitch_hi=55,  # G3
            max_phrases_per_section=1,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"part_dropped:bass:render_error:{exc}")
        bass_pm = _empty_pm(bpm)
        bass_pm.instruments.append(
            pretty_midi.Instrument(program=PROG_BASS, is_drum=False, name="bass")
        )

    try:
        harmony_pm = _render_harmony(plan, bpm, song_bars, warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"part_dropped:harmony:render_error:{exc}")
        harmony_pm = _empty_pm(bpm)
        harmony_pm.instruments.append(
            pretty_midi.Instrument(program=PROG_HARMONY, is_drum=False, name="harmony")
        )

    try:
        melody_pm = _render_phrase_part(
            "melody",
            plan,
            parts.get("melody") or [],
            bpm,
            song_bars,
            warnings,
            program=PROG_MELODY,
            pitch_lo=60,  # C4
            pitch_hi=84,  # C6
            max_phrases_per_section=2,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"part_dropped:melody:render_error:{exc}")
        melody_pm = _empty_pm(bpm)
        melody_pm.instruments.append(
            pretty_midi.Instrument(program=PROG_MELODY, is_drum=False, name="melody")
        )

    for name, pm in (
        ("drums", drums_pm),
        ("bass", bass_pm),
        ("harmony", harmony_pm),
        ("melody", melody_pm),
    ):
        path = midi_dir / f"{name}.mid"
        pm.write(str(path))
        program = {
            "drums": 0,
            "bass": PROG_BASS,
            "harmony": PROG_HARMONY,
            "melody": PROG_MELODY,
        }[name]
        _finalize_midi_file(
            path,
            song_bars,
            expected_tracks=[(name, program, name == "drums")],
        )

    # merged song.mid
    song = _empty_pm(bpm)
    for pm in (drums_pm, bass_pm, harmony_pm, melody_pm):
        for inst in pm.instruments:
            # copy instrument
            new_inst = pretty_midi.Instrument(
                program=inst.program, is_drum=inst.is_drum, name=inst.name
            )
            for n in inst.notes:
                new_inst.notes.append(
                    pretty_midi.Note(n.velocity, n.pitch, n.start, n.end)
                )
            song.instruments.append(new_inst)
    song_path = out_dir / "song.mid"
    song.write(str(song_path))
    _finalize_midi_file(
        song_path,
        song_bars,
        expected_tracks=[
            ("drums", 0, True),
            ("bass", PROG_BASS, False),
            ("harmony", PROG_HARMONY, False),
            ("melody", PROG_MELODY, False),
        ],
    )

    # de-dupe warnings preserve order
    seen = set()
    uniq = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq
