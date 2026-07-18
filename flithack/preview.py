"""FluidSynth MIDI → preview.wav. Preview failure never fails the run."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_soundfont() -> Path | None:
    """Resolve SOUNDFONT_PATH relative to repo root; verify exists."""
    raw = os.getenv("SOUNDFONT_PATH", "assets/FluidR3_GM.sf2").strip() or "assets/FluidR3_GM.sf2"
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if p.is_file():
        return p
    # common Homebrew fallbacks
    candidates = [
        Path("/opt/homebrew/Cellar/fluid-synth/2.5.4/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2"),
        Path("/opt/homebrew/share/soundfonts/FluidR3_GM.sf2"),
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    ]
    # scan fluid-synth cellar
    cellar = Path("/opt/homebrew/Cellar/fluid-synth")
    if cellar.is_dir():
        candidates.extend(sorted(cellar.glob("*/share/fluid-synth/sf2/*.sf2")))
    for c in candidates:
        if c.is_file():
            return c
    return None


def render_preview(
    song_mid: str | Path,
    preview_wav: str | Path,
    *,
    sample_rate: int = 44100,
) -> tuple[bool, str | None]:
    """
    fluidsynth -ni <sf2> song.mid -F preview.wav -r 44100

    Returns (ok, warning_or_none). Never raises for missing tools.
    """
    song_mid = Path(song_mid)
    preview_wav = Path(preview_wav)
    if not song_mid.is_file():
        return False, "preview_unavailable:missing_song_mid"

    fs = shutil.which("fluidsynth")
    if not fs:
        return False, "preview_unavailable:fluidsynth_not_found"

    sf = resolve_soundfont()
    if sf is None:
        return False, "preview_unavailable:soundfont_missing"

    preview_wav.parent.mkdir(parents=True, exist_ok=True)
    # fluidsynth 2.x: options first, then soundfont(s), then midifile(s)
    # e.g. fluidsynth -ni -F out.wav -r 44100 soundfont.sf2 song.mid
    cmd = [
        fs,
        "-ni",
        "-F",
        str(preview_wav),
        "-r",
        str(sample_rate),
        str(sf),
        str(song_mid),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        return False, f"preview_unavailable:{exc}"

    if proc.returncode != 0 or not preview_wav.is_file() or preview_wav.stat().st_size == 0:
        err = (proc.stderr or proc.stdout or "fluidsynth_failed")[-300:]
        return False, f"preview_unavailable:{err}"
    return True, None
