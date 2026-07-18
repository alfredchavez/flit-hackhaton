# flithack

Hackathon pipeline: **reference audio → musical profile + editable MIDI parts**, plus optional LLM interpretation and a Streamlit UI.

```bash
# pyenv virtualenv flithack-3.11 (see .python-version)
python -m flithack analyze reference.mp3 -o analysis_output/
python -m flithack interpret analysis_output/
streamlit run app.py
```

Output contract: `SPEC.md` §4 + additive `llm_interpretation.json` (`SPEC_2.md`).

## Setup

```bash
brew install ffmpeg
pyenv local flithack-3.11
pip install -e .
pip install "git+https://github.com/xavriley/ADTOF-pytorch"   # drums
pip install streamlit openai python-dotenv pydantic            # UI + LLM (pinned in requirements.txt)

cp .env.example .env   # set OPENAI_API_KEY
```

## Commands

```bash
# Full pipeline (stages 1–8)
python -m flithack analyze song.mp3 -o analysis_output/
python -m flithack analyze song.mp3 -o analysis_output/ --force
python -m flithack analyze song.mp3 -o analysis_output/ --no-quantize

# MIDI → text only (no API)
python -m flithack interpret analysis_output/ --dump-repr

# LLM interpretation on cached analysis (stages 9–10)
python -m flithack interpret analysis_output/
python -m flithack interpret analysis_output/ --force

# Block B: new track from analysis_output/ + prompt (stages 11–14)
python -m flithack generate analysis_output/ -p "boss battle, darker, faster"
python -m flithack generate analysis_output/ -p "calm exploration" -o generation_output/

# Render hardcoded fixture (no LLM) — hour-0 gate
python -m flithack generate --render-fixture tests/fixtures/generation_clean.json -o /tmp/gen_fix

# Fake fixture for downstream
python -m flithack fixture -o analysis_output/

# Web UI (imports stage functions; not a CLI subprocess)
streamlit run app.py
```

Optional: `brew install fluid-synth` and place `assets/FluidR3_GM.sf2` (or set `SOUNDFONT_PATH`). Preview failure still leaves MIDI intact.

## Stages

1. Ingest (`ffmpeg` → 44.1 kHz WAV)
2. Timeline (first downbeat → musical t=0)
3. Stem separation (`demucs-mlx` / `audio-separator`)
4. Global analysis (BPM, key, energy, sections)
5. Chords (beat-synchronous chroma)
6. Transcription (ADTOF drums + Basic Pitch)
7. Postprocess + melody
8. Package `analysis_output/`
9. MIDI → text (`midi_repr.py`)
10. LLM interpretation (`interpret.py`, OpenAI mini) — never fails the pipeline
11. PLAN (LLM → key/bpm/sections/chords)
12. PARTS (LLM → drum grids + bass/melody phrases)
13. RENDER (deterministic MIDI)
14. PREVIEW (FluidSynth → preview.wav, optional)
