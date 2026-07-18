"""Stem type → raw MIDI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

# Spec GM drum map
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_CLOSED_HAT = 42
DRUM_TOM = 47
DRUM_CYMBAL = 49

# ADTOF uses 35 for kick; remap to spec 36
_ADTOF_REMAP = {35: DRUM_KICK}


def _set_tempo(pm: pretty_midi.PrettyMIDI, bpm: float) -> pretty_midi.PrettyMIDI:
    """Embed a single global tempo; rebuild if needed."""
    # pretty_midi stores tempo changes; write a clean file with tempo at 0.
    out = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program,
            is_drum=inst.is_drum,
            name=inst.name,
        )
        for n in inst.notes:
            new_inst.notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start,
                    end=n.end,
                )
            )
        out.instruments.append(new_inst)
    return out


def _transcribe_basic_pitch(stem_path: Path, bpm: float) -> pretty_midi.PrettyMIDI:
    from basic_pitch.inference import predict

    _model_out, midi_data, _notes = predict(str(stem_path), midi_tempo=float(bpm))
    return _set_tempo(midi_data, bpm)


def _transcribe_drums_adtof(stem_path: Path, out_midi: Path, bpm: float) -> pretty_midi.PrettyMIDI:
    from adtof_pytorch import transcribe_to_midi

    tmp = out_midi.with_suffix(".adtof_raw.mid")
    transcribe_to_midi(str(stem_path), str(tmp), device="cpu")
    pm = pretty_midi.PrettyMIDI(str(tmp))
    # Remap pitches + force drum channel conventions via is_drum
    out = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for inst in pm.instruments:
        for n in inst.notes:
            pitch = _ADTOF_REMAP.get(n.pitch, n.pitch)
            # Only keep our five classes if possible; keep others that are close.
            if pitch not in (
                DRUM_KICK,
                DRUM_SNARE,
                DRUM_CLOSED_HAT,
                DRUM_TOM,
                DRUM_CYMBAL,
                35,
                36,
                37,
                38,
                40,
                41,
                42,
                43,
                45,
                46,
                47,
                49,
                51,
            ):
                continue
            if pitch == 35:
                pitch = DRUM_KICK
            drum.notes.append(
                pretty_midi.Note(
                    velocity=max(1, min(127, n.velocity)),
                    pitch=int(pitch),
                    start=float(n.start),
                    end=max(float(n.start) + 0.05, float(n.end)),
                )
            )
    out.instruments.append(drum)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return out


def _transcribe_drums_librosa(stem_path: Path, bpm: float) -> pretty_midi.PrettyMIDI:
    """Fallback: onset detection + crude frequency-band classification."""
    import librosa

    y, sr = librosa.load(str(stem_path), sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames", backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    hop = 512
    # Band energies around each onset.
    S = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)

    def band_energy(frame: int, fmin: float, fmax: float) -> float:
        mask = (freqs >= fmin) & (freqs < fmax)
        if frame >= S.shape[1]:
            return 0.0
        return float(np.mean(S[mask, frame])) if np.any(mask) else 0.0

    out = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for t, fr in zip(onset_times, onset_frames):
        fr = int(fr)
        low = band_energy(fr, 20, 150)
        mid = band_energy(fr, 150, 500)
        high = band_energy(fr, 500, 2000)
        hat = band_energy(fr, 2000, 8000)
        scores = {
            DRUM_KICK: low,
            DRUM_SNARE: mid,
            DRUM_TOM: (low + mid) * 0.5,
            DRUM_CLOSED_HAT: hat,
            DRUM_CYMBAL: hat * 0.7 + high * 0.3,
        }
        pitch = max(scores, key=scores.get)
        drum.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=float(t), end=float(t) + 0.08)
        )
    out.instruments.append(drum)
    return out


def transcribe_stem(
    stem_path: Path,
    stem_type: str,
    out_midi: Path,
    *,
    bpm: float = 120.0,
    force: bool = False,
) -> Path:
    """
    Transcribe one stem to raw MIDI.

    stem_type: drums | bass | vocals | other
    Public stage entrypoint.
    """
    stem_path = Path(stem_path)
    out_midi = Path(out_midi)
    out_midi.parent.mkdir(parents=True, exist_ok=True)

    from flithack.cache import stage_complete, write_marker

    options = {"stem_type": stem_type, "bpm": round(float(bpm), 3), "version": 1}
    if stage_complete(
        out_midi.parent,
        f"transcribe_{stem_type}",
        source=stem_path,
        options=options,
        expected_outputs=[out_midi],
        force=force,
    ):
        return out_midi

    if not stem_path.is_file():
        raise FileNotFoundError(f"stem not found: {stem_path}")

    print(f"[transcribe] {stem_type}: {stem_path.name}")
    if stem_type == "drums":
        pm = None
        try:
            pm = _transcribe_drums_adtof(stem_path, out_midi, bpm)
            n_notes = sum(len(i.notes) for i in pm.instruments)
            if n_notes < 4:
                print(f"[transcribe] ADTOF too sparse ({n_notes} notes); librosa drum fallback")
                pm = None
        except Exception as exc:  # noqa: BLE001
            print(f"[transcribe] ADTOF failed ({exc}); librosa drum fallback")
            pm = None
        if pm is None:
            pm = _transcribe_drums_librosa(stem_path, bpm)
    else:
        try:
            pm = _transcribe_basic_pitch(stem_path, bpm)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"basic-pitch failed on {stem_type}: {exc}") from exc
        # Name the instrument track
        if pm.instruments:
            pm.instruments[0].name = stem_type
            pm.instruments[0].is_drum = False
        else:
            pm.instruments.append(
                pretty_midi.Instrument(program=0, is_drum=False, name=stem_type)
            )

    pm = _set_tempo(pm, bpm)
    # Ensure single instrument track
    if len(pm.instruments) > 1:
        merged = pretty_midi.Instrument(
            program=pm.instruments[0].program,
            is_drum=pm.instruments[0].is_drum,
            name=stem_type,
        )
        for inst in pm.instruments:
            merged.notes.extend(inst.notes)
        merged.notes.sort(key=lambda n: (n.start, n.pitch))
        pm.instruments = [merged]
    elif pm.instruments:
        pm.instruments[0].name = stem_type

    pm.write(str(out_midi))
    if out_midi.stat().st_size == 0:
        raise RuntimeError(f"transcription wrote empty MIDI: {out_midi}")

    write_marker(
        out_midi.parent,
        f"transcribe_{stem_type}",
        source=stem_path,
        options=options,
        outputs=[out_midi],
    )
    return out_midi


def empty_midi(out_midi: Path, bpm: float, *, is_drum: bool = False, name: str = "") -> Path:
    """Write a valid empty MIDI with tempo (for sparse vocals etc.). Still a real file."""
    out_midi = Path(out_midi)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    pm.instruments.append(pretty_midi.Instrument(program=0, is_drum=is_drum, name=name))
    pm.write(str(out_midi))
    return out_midi
