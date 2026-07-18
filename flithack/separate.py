"""Audio → stems (drums / bass / vocals / other)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

STEM_NAMES = ("drums", "bass", "vocals", "other")


def _write_stem(path: Path, wav: np.ndarray, samplerate: int) -> None:
    """wav: (channels, samples) or (samples, channels) or 1d."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(wav)
    if arr.ndim == 1:
        data = arr
    elif arr.shape[0] <= 8 and arr.shape[0] < arr.shape[-1]:
        # (channels, samples)
        data = arr.T
    else:
        data = arr
    sf.write(str(path), data, samplerate)


def _separate_demucs_mlx(aligned_wav: Path, stems_dir: Path) -> dict[str, Path]:
    """Primary path. Avoids mlx_audio_io load/save when ABI is broken — use soundfile I/O."""
    from demucs_mlx import Separator

    sep = Separator(model="htdemucs", progress=True)
    # Load with soundfile (mlx_audio_io often ABI-mismatches mlx on pyenv envs).
    data, file_sr = sf.read(str(aligned_wav), always_2d=True, dtype="float32")
    # data: (samples, channels) → (channels, samples)
    wav = data.T
    target_sr = int(sep.samplerate)
    if file_sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=target_sr, axis=-1)

    _mix, stems = sep.separate_tensor(wav)
    sr = target_sr
    out: dict[str, Path] = {}
    for name in STEM_NAMES:
        if name not in stems:
            raise RuntimeError(f"demucs-mlx missing stem '{name}'; got {list(stems)}")
        dest = stems_dir / f"{name}.wav"
        _write_stem(dest, stems[name], sr)
        out[name] = dest
    return out


def _separate_audio_separator(aligned_wav: Path, stems_dir: Path) -> dict[str, Path]:
    """Fallback: audio-separator with htdemucs."""
    from audio_separator.separator import Separator

    stems_dir.mkdir(parents=True, exist_ok=True)
    sep = Separator(
        output_dir=str(stems_dir),
        output_format="WAV",
    )
    # Model name conventions vary; try common ones.
    model_candidates = [
        "htdemucs.yaml",
        "htdemucs",
        "UVR-MDX-NET-Inst_HQ_3.onnx",
    ]
    last_err: Exception | None = None
    output_files = None
    for model in model_candidates:
        try:
            sep.load_model(model_filename=model)
            output_files = sep.separate(str(aligned_wav))
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if output_files is None:
        raise RuntimeError(f"audio-separator failed: {last_err}")

    # Resolve returned paths (may be bare filenames) and scan output dir.
    resolved: list[Path] = []
    for p in output_files or []:
        path = Path(p)
        if not path.is_file():
            cand = stems_dir / path.name
            if cand.is_file():
                path = cand
        if path.is_file():
            resolved.append(path)
    candidates = {p.resolve(): p for p in resolved}
    for p in stems_dir.glob("*.wav"):
        candidates[p.resolve()] = p

    def find_stem(key: str) -> Path | None:
        key_l = key.lower()
        for p in candidates.values():
            if key_l in p.name.lower():
                return p
        return None

    out: dict[str, Path] = {}
    for name in STEM_NAMES:
        found = find_stem(name)
        if found is None:
            names = [p.name for p in candidates.values()]
            raise RuntimeError(f"audio-separator missing stem '{name}'; files={names}")
        dest = stems_dir / f"{name}.wav"
        if found.resolve() != dest.resolve():
            data, sr = sf.read(str(found.resolve()), always_2d=True)
            sf.write(str(dest), data, sr)
        out[name] = dest
    return out


def separate_stems(
    aligned_wav: Path,
    stems_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """
    Separate aligned mix into drums/bass/vocals/other WAV stems.

    Public stage entrypoint. Writes into stems_dir.
    """
    aligned_wav = Path(aligned_wav)
    stems_dir = Path(stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)

    from flithack.cache import stage_complete, write_marker

    options = {"model": "htdemucs", "version": 1}
    expected = [stems_dir / f"{n}.wav" for n in STEM_NAMES]
    if stage_complete(
        stems_dir,
        "separate",
        source=aligned_wav,
        options=options,
        expected_outputs=expected,
        force=force,
    ):
        return {n: stems_dir / f"{n}.wav" for n in STEM_NAMES}

    try:
        print("[separate] running demucs-mlx (htdemucs)…")
        out = _separate_demucs_mlx(aligned_wav, stems_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[separate] demucs-mlx failed ({exc}); falling back to audio-separator")
        out = _separate_audio_separator(aligned_wav, stems_dir)

    for name in STEM_NAMES:
        p = out[name]
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"stem separation produced empty/missing {name}: {p}")

    write_marker(
        stems_dir,
        "separate",
        source=aligned_wav,
        options=options,
        outputs=expected,
    )
    return out
