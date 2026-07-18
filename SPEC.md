# SPEC — Reference Audio → Analysis + Per-Part MIDI

Hackathon spec. Goal: **functional and dirty**.

Optimize for one reliable demo on macOS Apple Silicon, not production architecture, perfect transcription, or long-term dependency purity. Stable boundaries matter; elegance inside each stage does not.

## 1. Product idea

### User story

An indie game creator wants to build and sell a game but knows little or nothing about music. They can explain the mood they want and provide a reference track, but they cannot compose, arrange, or prepare adaptive game audio themselves.

### Product promise

**Prompt + reference audio → new, editable MIDI parts.**

The user can preview the result, import it into a DAW, change instruments or notes, and eventually render separate audio stems for the game.

Why MIDI:

- It remains editable instead of becoming one opaque generated WAV.
- It is a standard interchange format supported by DAWs and music tooling.
- Separate drum, bass, melody, and harmony parts can later be rendered as adaptive game-audio stems.
- A musician can improve the result later without starting over.

The positioning is not “we produce more polished final audio than Suno or ElevenLabs.” It is:

> We turn musical intent and a reference into editable building blocks that can become game-ready adaptive audio.

### Whole-product flow

```text
user prompt + reference audio
              │
              ├──────────── prompt ─────────────────────┐
              │                                         │
              ▼                                         ▼
[A] reference analysis — THIS REPO             [B] new MIDI generation
    audio → musical profile + MIDI parts           prompt + reference profile
              │                                    + reference MIDI parts
              └──────── analysis_output/ ───────────┘
                                                        │
                                                        ▼
                                            new editable MIDI parts
                                                        │
                              ┌─────────────────────────┼────────────────────┐
                              ▼                         ▼                    ▼
                       browser preview            export to DAW      MIDI → instruments
                       (implementation TBD)                            → audio stems
```

The web app owns upload, prompt entry, progress, and presentation. Browser MIDI playback may use Magenta.js, a SoundFont player, or another synth, but that choice is downstream and must not affect this repo’s contract.

## 2. Scope of this repo

This repo implements **only block A**:

```text
reference audio (mp3/wav)
    ↓
[1] ingest                 → normalized full-length WAV
[2] timeline alignment     → first reliable downbeat becomes musical t=0
[3] stem separation        → drums / bass / vocals / other (.wav)
[4] global analysis        → BPM, beats, downbeats, key, meter, energy, sections
[5] chord estimation       → coarse chord timeline
[6] stem transcription     → per-part MIDI
[7] melody extraction      → cleaned melody.mid
[8] package                → analysis_output/
```

We stop when the reference has been **coarsely analyzed and transcribed into editable MIDI parts**.

This repo does **not**:

- Interpret the user prompt.
- Generate new music.
- Preview MIDI in the browser.
- Choose final instruments.
- Place MIDI in a DAW.
- Render the final MIDI to audio.

The downstream generator receives two inputs independently:

1. The user’s prompt from the web app.
2. `analysis_output/` from this repo.

The only interface exposed by this repo is the `analysis_output/` folder. The downstream teammate must be able to develop from a hand-written fake folder from hour zero.

## 3. Hackathon success criteria

The MVP succeeds when one command:

```bash
python -m flithack analyze reference.mp3 -o analysis_output/
```

produces an output that the downstream half can consume without manual repair.

Required demo checks:

- Four reference audio stems exist and their `t=0` is the first musical downbeat.
- `reference_profile.json` contains usable timing, key, and energy information, plus chords or an explicit chord warning. Coarse sections are optional hints.
- Drum, bass, vocal, other, and melody MIDI files can be opened.
- The MIDI files share the same tempo and musical time origin, so bar 1 beat 1 lands at DAW time `0`.
- Drums, bass, and melody are recognizably related to the reference when played with a basic synth.
- The downstream teammate can replace their fake fixture with a real `analysis_output/` without changing their parser.

Quality target: **recognizable and structurally useful**, not note-perfect.

## 4. Output contract

Freeze this contract before implementation. Adding optional JSON fields is allowed; renaming or changing the meaning of existing fields requires agreement with the downstream teammate.

```text
analysis_output/
├── reference_profile.json
├── stems/
│   ├── drums.wav
│   ├── bass.wav
│   ├── vocals.wav
│   └── other.wav
└── midi/
    ├── drums.mid
    ├── bass.mid
    ├── vocals.mid
    ├── other.mid
    └── melody.mid
```

### Artifact meaning

| Artifact | Meaning | MVP status |
|---|---|---|
| `drums.mid` | Five-class drum transcription on the GM drum channel | Required |
| `bass.mid` | Mostly monophonic bass transcription | Required |
| `vocals.mid` | Raw pitched transcription of the vocal stem | Required, may be sparse |
| `other.mid` | Polyphonic harmonic sketch; expected to be messy | Required, low quality accepted |
| `melody.mid` | Cleaned monophonic melody source | Required |

For a vocal track, `melody.mid` is a cleaned version of `vocals.mid`. If the vocal stem has too few pitched notes, derive a crude melody from `other.mid` by keeping the most prominent upper-register line. This fallback only needs to work for the selected demo tracks.

### `reference_profile.json` v0.1

```json
{
  "schema_version": "0.1",
  "source_file": "reference.mp3",
  "source_offset_seconds": 0.51,
  "timeline_origin": "first_downbeat",
  "duration_seconds": 212.89,
  "bpm": 118.2,
  "meter": "4/4",
  "key": "F minor",
  "beats": [0.0, 0.51, 1.02],
  "downbeats": [0.0, 2.03, 4.06],
  "energy_curve": [
    {"start": 0.0, "end": 2.03, "value": 0.12},
    {"start": 2.03, "end": 4.06, "value": 0.35}
  ],
  "sections": [
    {"start": 0.0, "end": 16.24, "energy": "low"},
    {"start": 16.24, "end": 48.72, "energy": "medium"},
    {"start": 48.72, "end": 64.96, "energy": "high"}
  ],
  "chords": [
    {"start": 0.0, "end": 2.03, "chord": "Fm"},
    {"start": 2.03, "end": 4.06, "chord": "Ab"}
  ],
  "per_stem": {
    "drums": {"onsets_per_bar": 10.5},
    "bass": {
      "notes_per_bar": 3.2,
      "register_low_midi": 36,
      "register_high_midi": 55
    },
    "melody": {
      "notes_per_bar": 4.1,
      "range_semitones": 9,
      "median_pitch_midi": 62
    }
  },
  "warnings": []
}
```

Field rules:

- Detect the first reliable downbeat on the full normalized mix and trim the mix there before separation. All derived stems inherit that aligned `t=0`; subtract the same offset from every analysis timestamp and MIDI event.
- `source_offset_seconds` records how many seconds were removed from the original source before musical `t=0`.
- `timeline_origin` is `first_downbeat` when alignment succeeds. If it fails, retain the original start, set it to `source_start`, and emit `timeline_unaligned`.
- All contract times are seconds from the aligned musical `t=0`, not from the original uploaded file.
- `duration_seconds` is the duration after timeline alignment.
- `bpm` is one global tempo. Tempo-changing music is unsupported in the MVP.
- `energy_curve` contains one entry per detected bar.
- Energy is bar RMS divided by the loudest bar RMS in the same track, producing `0.0–1.0` values.
- `sections` is an optional hint owned by this repo: smooth bar energy, split on large sustained changes, merge segments shorter than four bars, and label segments `low`, `medium`, or `high` relative to this track. Downstream must not treat these labels as semantic claims such as verse or chorus.
- `chords` may be empty when estimation is unreliable, but `warnings` must then contain `chords_unreliable`; downstream owns the key-only fallback progression.
- Density values are raw average event counts per detected bar, not arbitrary normalized scores.
- MIDI pitch values use the standard `0–127` note-number range.
- `warnings` records usable fallbacks such as `timeline_unaligned`, `meter_assumed_4_4`, `key_low_confidence`, `chords_unreliable`, or `melody_from_other`.
- A fatal stage failure fails the command; it must not silently produce an empty required artifact.

### MIDI conventions

- Every file embeds the detected tempo and begins on the same aligned timeline as the WAV stems.
- Musical `t=0` is bar 1 beat 1: the first reliable downbeat, not merely the beginning of the uploaded file.
- Quantize notes to a 1/16 grid derived from the detected beat positions.
- Keep `--no-quantize` for debugging.
- Drop notes shorter than 60 ms after transcription.
- Use one instrument track per file.
- Drums use General MIDI percussion on human-numbered channel 10, which is zero-based channel 9 in MIDI code.
- Drum mapping: kick `36`, snare `38`, closed hat `42`, tom `47`, cymbal `49`.
- Bass, vocals, and melody should be monophonic-ish. `other.mid` may be polyphonic.

## 5. Tool choices

These are the weekend defaults. The fallback list is a contingency plan, not an instruction to implement every adapter in advance.

| Stage | Primary | Implement only if primary fails |
|---|---|---|
| Ingest | `ffmpeg` → 44.1 kHz WAV | No fallback |
| Stem separation | `demucs-mlx` | `audio-separator` with `htdemucs`; then hosted Demucs |
| Beats/downbeats | `beat-this` | `librosa.beat.beat_track`; assume `4/4` and emit warning |
| BPM | `60 / median(inter-beat interval)` | Same librosa fallback |
| Key | librosa chroma + Krumhansl-Schmuckler templates | Essentia only if already installable |
| Chords | Beat-synchronous chroma + major/minor templates | Emit `chords_unreliable` and let downstream generate from key; no model upgrade unless downstream is blocked |
| Bass/vocals/other → MIDI | `basic-pitch` | Separate Python 3.10 Basic Pitch runner; then `basic-pitch-torch` |
| Drums → MIDI | `ADTOF-pytorch` | Librosa onsets + crude frequency-band classification |
| Melody cleanup | Dirty postprocessing of vocals; fall back to other | No advanced melody model in MVP |
| MIDI I/O | `pretty_midi` | `mido` |

Rules:

- Timebox a fighting dependency to 20 minutes.
- Preserve the stage function signature when replacing a tool.
- Do not build fallback implementations until the primary has actually failed.
- A separate runtime invoked through a subprocess is acceptable. Keeping everything in one Python environment is not a product requirement.
- License debt is acceptable for a private hackathon demo only. Re-audit models and dependencies before any commercial use.

## 6. Environment and dependency spike

Main environment: Python 3.11 virtual environment `flithack-3.11` on macOS Apple Silicon.

Likely friction:

1. Basic Pitch documents general Python 3.11 support but its Apple Silicon path is documented around Python 3.10. If the main environment fails, create a small Python 3.10 transcription runner and call it through files/subprocesses.
2. `demucs-mlx` is young. Fall back quickly if installation or output naming is unstable.
3. `ADTOF-pytorch` is a small Git-installed project. Do not spend the day repairing it.

Initial attempt:

```bash
brew install ffmpeg
pip install demucs-mlx beat-this basic-pitch librosa pretty_midi soundfile
pip install git+https://github.com/xavriley/ADTOF-pytorch
```

Before writing the application structure, run every risky dependency against the same 20–30 second WAV clip. The spike passes only when it produces:

- Four separated WAV files.
- Beat and downbeat arrays.
- At least one Basic Pitch MIDI.
- A drum MIDI from either the primary or fallback path.

Record the working commands immediately. Those commands become the implementation; do not re-research tools during the weekend.

## 7. Code shape

One package, one CLI, no server:

```text
flithack/
├── __init__.py
├── __main__.py       # enables: python -m flithack
├── cli.py
├── timeline.py       # beat/downbeat prepass; trim and shift to musical t=0
├── separate.py       # audio → stems
├── analyze.py        # BPM, beats, key, meter, energy, coarse sections
├── chords.py         # coarse chord timeline
├── transcribe.py     # stem type → raw MIDI
├── postprocess.py    # quantize, filter, melody extraction/cleanup
└── package.py        # validate and package the output contract
```

Each stage exposes one public function with paths in and paths/data out. The orchestrator may call those functions, but a stage must not depend on another stage’s implementation details.

Keep it dirty inside a stage. A 200-line `chords.py` is acceptable. `transcribe.py` branching on which separator happened to run is not.

### Cheap, safe caching

Each stage writes a completion marker only after all its expected outputs exist. The marker stores:

- Source filename, size, and modification time.
- Stage name.
- Relevant options.
- Expected output filenames.

Reuse a stage only when its marker still matches. `--force` ignores markers. An existing output file without a valid marker is incomplete and must be regenerated.

## 8. Build order and gates

### 0. Freeze the boundary

- Hand the downstream teammate a fake `analysis_output/` matching section 4.
- Confirm which exact fields and MIDI files their first generator consumes.

Gate: their code reads the fixture successfully.

### 1. Dependency spike

- Run separation, beat tracking, Basic Pitch, and drum transcription on one 20–30 second clip.
- Select a fallback immediately when a timebox expires.

Gate: every high-risk stage has a known working command or chosen fallback.

### 2. Thin end-to-end vertical slice

- Wire one command through ingest, timeline alignment, separation, minimal analysis, all MIDI outputs, and packaging.
- Hardcoded values are acceptable temporarily where the dependency spike has not yet been integrated.

Gate: a real `analysis_output/` replaces the teammate’s fake fixture and their parser still works.

### 3. Make the musical profile real

- Integrate beats, downbeats, BPM, meter, key, energy, optional coarse sections, and coarse chords.

Gate: BPM is within ±2 of manual tapping. Key and simple-pop chords sound plausible, or the profile emits `chords_unreliable` and the downstream generator successfully falls back to a key-only progression.

### 4. Make the MIDI recognizable

- Integrate per-stem transcription.
- Verify the shared `t=0`, quantize, remove glitches, and produce `melody.mid`.

Gate: drums, bass, and melody sound recognizably related to the reference over a basic synth.

### 5. Integration demo

- Run the primary demo song end to end.
- Pass the result to the new-MIDI generator.
- Preview or import the generated MIDI downstream.

Gate: prompt + reference produce new editable MIDI parts. The demo must not end at “we created JSON.”

### 6. Stretch only

- Run a second contrasting song.
- Improve vocal/melody cleanup.
- Tune quantization.
- Improve `other.mid`.

## 9. Demo material

Get one easy song working before testing contrast.

Primary reference requirements:

- Steady tempo.
- `4/4` meter.
- Simple pop, rock, or electronic harmony.
- Clear bass.
- Clear lead vocal or instrumental melody.
- A clear first downbeat with no important pickup notes before it.
- No rubato, meter changes, dense jazz harmony, or long ambient intro.

After the primary reference works, add one contrasting reference: slow/sparse/dark versus fast/bright/dense. Different references should create visibly and audibly different profiles and downstream MIDI.

Prefer owned, licensed, or public-domain demo material so the pitch does not get derailed by a copying question. Originality detection is still outside the build scope.

## 10. Explicitly not doing

- No web UI, server, queue, or job orchestration in this repo.
- No prompt interpretation or new-song generation in this repo.
- No browser player or commitment to Magenta.js in this repo.
- No final instrument selection or audio rendering.
- No perfect transcription.
- No reliable support for changing tempo or meter.
- No detailed drum taxonomy beyond five classes.
- No instrument identification inside `other`.
- No copyright filter or originality score.
- No model benchmarking.
- No cleanup refactor unless it directly fixes the demo.

## 11. Final demo sentence

> Give us a musical idea and a reference. We extract its rhythm, harmony, energy, and editable parts, then use that structure to create new MIDI you can preview, modify, and take into your DAW instead of being trapped inside one generated audio file.
