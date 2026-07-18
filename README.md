# flithack

Hackathon **block A**: reference audio → musical profile + editable MIDI parts.

```bash
# pyenv virtualenv flithack-3.11 (see .python-version)
python -m flithack analyze reference.mp3 -o analysis_output/
```

Output contract: see `SPEC.md` §4 (`analysis_output/`).

## Setup

```bash
brew install ffmpeg
pyenv local flithack-3.11   # or: pyenv activate flithack-3.11
pip install -e .
# optional drum model (if not already installed)
pip install "git+https://github.com/xavriley/ADTOF-pytorch"
```

## Commands

```bash
# Full pipeline
python -m flithack analyze song.mp3 -o analysis_output/

# Ignore stage caches
python -m flithack analyze song.mp3 -o analysis_output/ --force

# Skip 1/16 quantization (debug)
python -m flithack analyze song.mp3 -o analysis_output/ --no-quantize

# Fake fixture for downstream teammate
python -m flithack fixture -o analysis_output/
```

## Stages

1. Ingest (`ffmpeg` → 44.1 kHz WAV)
2. Timeline (first downbeat → musical t=0)
3. Stem separation (`demucs-mlx` / `audio-separator`)
4. Global analysis (BPM, key, energy, sections)
5. Chords (beat-synchronous chroma)
6. Transcription (ADTOF drums + Basic Pitch)
7. Postprocess + melody
8. Package `analysis_output/`
