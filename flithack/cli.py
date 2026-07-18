"""CLI: python -m flithack analyze reference.mp3 -o analysis_output/"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ingest(input_path: Path, out_wav: Path, sample_rate: int = 44100) -> Path:
    """ffmpeg → normalized full-length WAV."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-sample_fmt",
        "s16",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")
    if not out_wav.is_file() or out_wav.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced empty output")
    return out_wav


def cmd_analyze(args: argparse.Namespace) -> int:
    from flithack.analyze import analyze_global, compute_per_stem_stats
    from flithack.chords import estimate_chords
    from flithack.package import build_profile, package_output
    from flithack.postprocess import build_melody, postprocess_part
    from flithack.separate import separate_stems
    from flithack.timeline import align_timeline
    from flithack.transcribe import transcribe_stem

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    force = bool(args.force)
    quantize = not bool(args.no_quantize)

    print(f"[flithack] input={input_path}")
    print(f"[flithack] output={output_dir}")
    print(f"[flithack] work={work_dir}")

    # [1] ingest
    normalized = work_dir / "normalized.wav"
    if force or not normalized.is_file():
        print("[1/8] ingest (ffmpeg)…")
        ingest(input_path, normalized)
    else:
        print("[1/8] ingest (cached)")

    # [2] timeline alignment
    print("[2/8] timeline alignment…")
    timeline = align_timeline(normalized, work_dir / "timeline", force=force)
    print(
        f"       offset={timeline.source_offset_seconds:.3f}s "
        f"origin={timeline.timeline_origin} bpm={timeline.bpm:.2f}"
    )

    # [3] stem separation
    print("[3/8] stem separation…")
    stems_work = work_dir / "stems"
    stem_paths = separate_stems(timeline.aligned_wav, stems_work, force=force)

    # [4] global analysis
    print("[4/8] global analysis…")
    analysis = analyze_global(
        timeline.aligned_wav,
        beats=timeline.beats,
        downbeats=timeline.downbeats,
        bpm=timeline.bpm,
        meter=timeline.meter,
        duration_seconds=timeline.duration_seconds,
        extra_warnings=list(timeline.warnings),
    )

    # [5] chords
    print("[5/8] chord estimation…")
    chords, chord_warnings = estimate_chords(
        timeline.aligned_wav,
        analysis.beats,
        duration_seconds=analysis.duration_seconds,
    )
    all_warnings = list(analysis.warnings) + list(chord_warnings)

    # [6] stem transcription → raw MIDI in work dir
    print("[6/8] stem transcription…")
    raw_midi_dir = work_dir / "midi_raw"
    raw_midi_dir.mkdir(parents=True, exist_ok=True)
    raw_paths: dict[str, Path] = {}
    for stem_type in ("drums", "bass", "vocals", "other"):
        raw_paths[stem_type] = transcribe_stem(
            stem_paths[stem_type],
            stem_type,
            raw_midi_dir / f"{stem_type}.mid",
            bpm=analysis.bpm,
            force=force,
        )

    # [7] postprocess + melody
    print("[7/8] postprocess + melody…")
    final_midi_dir = work_dir / "midi_final"
    final_midi_dir.mkdir(parents=True, exist_ok=True)
    midi_paths: dict[str, Path] = {}
    for part in ("drums", "bass", "vocals", "other"):
        midi_paths[part] = postprocess_part(
            raw_paths[part],
            final_midi_dir / f"{part}.mid",
            part=part,
            bpm=analysis.bpm,
            beats=analysis.beats,
            quantize=quantize,
            duration=analysis.duration_seconds,
        )

    melody_path, melody_warnings = build_melody(
        midi_paths["vocals"],
        midi_paths["other"],
        final_midi_dir / "melody.mid",
        bpm=analysis.bpm,
        beats=analysis.beats,
        quantize=quantize,
        duration=analysis.duration_seconds,
    )
    midi_paths["melody"] = melody_path
    all_warnings.extend(melody_warnings)

    # per-stem stats from final MIDI
    per_stem = compute_per_stem_stats(
        midi_paths,
        analysis.downbeats,
        analysis.duration_seconds,
    )

    # [8] package
    print("[8/8] package…")
    # Copy stems into output layout via package_output
    out_stems = output_dir / "stems"
    out_stems.mkdir(parents=True, exist_ok=True)
    profile = build_profile(
        source_file=input_path.name,
        source_offset_seconds=timeline.source_offset_seconds,
        timeline_origin=timeline.timeline_origin,
        duration_seconds=analysis.duration_seconds,
        bpm=analysis.bpm,
        meter=analysis.meter,
        key=analysis.key,
        beats=analysis.beats,
        downbeats=analysis.downbeats,
        energy_curve=analysis.energy_curve,
        sections=analysis.sections,
        chords=chords,
        per_stem=per_stem,
        warnings=all_warnings,
    )
    package_output(
        output_dir,
        profile=profile,
        stem_paths=stem_paths,
        midi_paths=midi_paths,
    )

    print(f"[flithack] done → {output_dir}")
    print(f"  bpm={profile['bpm']:.2f} key={profile['key']} meter={profile['meter']}")
    print(f"  warnings={profile['warnings']}")
    return 0


def cmd_fixture(args: argparse.Namespace) -> int:
    from flithack.package import write_fake_fixture

    out = Path(args.output).resolve()
    write_fake_fixture(out)
    print(f"[flithack] wrote fake fixture → {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flithack",
        description="Reference audio → analysis_output/ (musical profile + MIDI parts)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run full reference analysis pipeline")
    analyze.add_argument("input", help="Reference audio (mp3/wav/…)")
    analyze.add_argument(
        "-o",
        "--output",
        default="analysis_output",
        help="Output directory (default: analysis_output/)",
    )
    analyze.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for intermediates (default: <output>/.work)",
    )
    analyze.add_argument(
        "--force",
        action="store_true",
        help="Ignore stage cache markers and recompute",
    )
    analyze.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip 1/16 quantization (debug)",
    )
    analyze.set_defaults(func=cmd_analyze)

    fixture = sub.add_parser(
        "fixture",
        help="Write a hand-written fake analysis_output/ for downstream",
    )
    fixture.add_argument(
        "-o",
        "--output",
        default="analysis_output",
        help="Output directory (default: analysis_output/)",
    )
    fixture.set_defaults(func=cmd_fixture)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            raise
        return 1
