# SPEC_3 — New-Track Generation (Block B)

Extends `SPEC.md` and `SPEC_2.md`. Same rules: **functional and dirty**, stable boundaries, elegance inside stages does not matter.

This is **block B** from SPEC.md §1: the analysis of the reference is done; now the AI treats it as *a reference only* and composes a **new** track in its musical language, output as downloadable MIDI parts.

Scope note: SPEC.md §2 said this repo "does not generate new music." That line is superseded the same way the UI ban was in SPEC_2. The boundary itself survives unchanged and is the whole point: **block B reads only `analysis_output/` + the user prompt.** It never touches the reference audio, the stems, or block A's internals. If block B works from a hand-faked `analysis_output/`, the boundary is being respected.

## 1. Flow

```text
analysis_output/  +  user prompt ("boss battle, darker, faster")
        │
        ▼
[11] PLAN call        LLM → generation_plan: key, bpm, sections, chords
        │
        ▼
[12] PARTS call       LLM → drum grids + bass/melody phrases per section
        │                   (same text language midi_repr uses — in reverse)
        ▼
[13] RENDER           deterministic: plan + patterns → valid MIDI files
        │
        ▼
[14] PREVIEW          FluidSynth → preview.wav
        │
        ▼
generation_output/    downloadable from the UI
```

**Core architecture rule (from the original architecture doc, non-negotiable):** the LLM chooses *musical decisions* — keys, chords, grids, phrases. It never emits raw MIDI events or thousands of notes. The renderer is deterministic Python that turns decisions into notes and **enforces validity**, so a sloppy LLM answer can degrade musicality but can never produce a broken file.

**Originality rule:** the system prompt must state that the analysis describes a *reference*, that the task is a new composition borrowing its musical language (tempo feel, chord colors, groove density, energy arc), and that exact melodies, basslines, and drum hooks from the reference must not be reproduced. The reference provides vocabulary, not content.

## 2. Output contract

```text
generation_output/
├── generation_plan.json     # schema + normalized plan + parts + warnings + provenance
├── generation_complete.json # written last; UI ignores runs without it
├── midi/
│   ├── drums.mid
│   ├── bass.mid
│   ├── harmony.mid
│   └── melody.mid
├── song.mid                 # all parts merged, one track each
└── preview.wav              # FluidSynth render (optional artifact, see §6)
```

- Same MIDI conventions as SPEC.md §4: shared tempo, t=0 at bar 1, 1/16 quantization, GM drums on channel 10 (kick 36, snare 38, closed hat 42, tom 47, cymbal 49).
- Default length: **16–32 bars, loop-friendly** — game music loops; a tight loop demos better than a meandering 3-minute form.
- In the UI, each generation run gets its own folder: `runs/<upload-sha>/generation_output/<n>/`. Regenerating never overwrites a previous result. Build in `<n>.partial/`, then atomically rename to `<n>/` only after every required MIDI exists and `generation_complete.json` has been written. The UI lists only numbered folders with that marker. A failed PLAN/PARTS call fails only the new generation attempt; block A and previous generations remain usable.
- `generation_plan.json` records: `schema_version: "0.1"`, user prompt, variation nonce, model, `prompt_sha256`, `analysis_sha256`, the full normalized plan/parts, and `warnings`. `prompt_sha256` covers both complete system prompts, both Pydantic schemas, and their prompt-version strings. `analysis_sha256` is SHA-256 over canonical JSON (`sort_keys=True`, compact separators) of exactly the profile and validated interpretation consumed, plus the exact `midi_repr()` string supplied to PARTS. Represent an absent interpretation as JSON `null`.
- `generation_complete.json` is the completion marker and records `schema_version`, every required filename (`generation_plan.json`, four part MIDIs, and `song.mid`), and their SHA-256 hashes. Run preview before writing the final plan/marker so `preview_unavailable` is persisted when needed. `preview.wav` is optional and is not required for completion.

## 3. The two LLM calls (`flithack/generate.py`)

Same SDK setup as SPEC_2 (`responses.parse` + Pydantic, key from env, 1 retry). Generation uses a 10-second connect timeout and a configurable read timeout from `GENERATION_TIMEOUT_SECONDS` (default `300` seconds), because structured PLAN/PARTS responses can exceed the interpretation call's 30-second budget. Model: `OPENAI_MODEL`; optional `GENERATION_MODEL` env override if mini-tier plans sound dumb and you want to bump only this half.

### Call 1 — PLAN

Input: user prompt + `reference_profile.json` (compact: bpm, key, meter, chord summary, energy arc, densities) + `llm_interpretation.json` only when its model/prompt/input provenance is still valid. Reuse `load_valid_cached_interpretation`; mere file presence is not validity.

```python
class Section(BaseModel):
    id: str                   # unique stable ID: intro / groove_a / groove_b / outro
    name: str                 # human label shown in the UI
    bars: int                 # 2–16
    energy: float             # 0–1
    active_parts: list[str]   # subset of: drums, bass, harmony, melody
    chords: list[str]         # one symbol per bar: "Fm", "Ab", "Cm7", "Bb"

class GenerationPlan(BaseModel):
    title: str
    bpm: float                # near reference unless the prompt says otherwise
    key: str                  # e.g. "C minor" — may differ from reference
    meter: str                # requested value; normalizer always forces "4/4"
    sections: list[Section]   # total 16–32 bars
    style_notes: str          # carried verbatim into the PARTS call
```

Immediately after parsing, run a deterministic `normalize_plan()` before PARTS or rendering. It:

- clamps BPM to `40–240`, section bars to `2–16`, and energy to `0–1`;
- parses the generated key; if unusable, falls back to the valid reference key or `C major` and warns;
- forces meter to `4/4` and warns `meter_forced_4_4` if needed;
- makes section IDs non-empty and unique, filters/deduplicates `active_parts` to the four allowed values, and preserves section order;
- pads/truncates each chord list to exactly the section's bar count, repeating the last valid chord or using the generated key's tonic triad when empty;
- creates a default 16-bar `groove_a` with all four parts and the tonic chord if no usable sections remain;
- deterministically pads/repeats or truncates the final form until the total is `16–32` bars; and
- records every repair in `warnings`.

The renderer receives only the normalized plan. A musically weak response may sound weak, but an out-of-range or malformed planning choice must not create a broken MIDI or crash the run.

### Call 2 — PARTS

Input: the normalized plan + the same compact style/interpretation context + the representative-bar `midi_repr()` text from block A. The prompt states that those bars are format and style evidence, not notes to copy. One call for all parts (fallback: one call per part only after the single call has actually failed the gate or is consistently truncated).

```python
class DrumPattern(BaseModel):
    section_id: str
    kick: str                 # exactly 16 chars of x / o / .   (x accent, o soft)
    snare: str
    closed_hat: str
    tom: str
    cymbal: str
    fill_last_bar: bool

class Phrase(BaseModel):
    section_id: str
    events_by_bar: list[str]  # 1–4 strings; renderer loops the list across section
                              # "F2@1.0 len1.0 accent | Ab2@3.5 len0.5 soft"

class Parts(BaseModel):
    drums: list[DrumPattern]  # one per section where drums are active
    bass: list[Phrase]        # one phrase per active section
    melody: list[Phrase]      # one or two per active section
```

Harmony needs **no LLM call**: the renderer builds it deterministically from the plan's chord symbols. One less thing the model can break.

The exact pitched-event grammar is:

```text
NOTE@BEAT lenDURATION VELOCITY [ | NOTE@BEAT lenDURATION VELOCITY ... ]
```

- `NOTE`: scientific pitch such as `F2`, `C#4`, or `Eb5`.
- `BEAT`: `1.0 <= beat < 5.0`, local to that bar.
- `DURATION`: positive beats.
- `VELOCITY`: exactly `accent`, `med`, or `soft`, matching `midi_repr`.
- One `events_by_bar` string represents one bar; the list length is the phrase length (`1–4` bars). An empty string is a rest bar.

This is the same text language `midi_repr` uses for *reading* the reference (SPEC_2 §3) — grids for drums, `pitch@beat` events for pitched parts. Reading and writing share one representation so prompt examples can literally show the reference's own bars as format examples. If two melody phrases target one section, the renderer alternates complete phrase cycles in list order; extra phrases are dropped with a warning. Missing patterns for an active part produce `part_dropped:<name>` rather than a fatal error.

Run a deterministic `normalize_parts()` before rendering: drop patterns for unknown section IDs; keep the first drum and bass pattern per section; keep at most the first two melody phrases per section; constrain each phrase to `1–4` bar strings; repair drum rows and event tokens under the forgiving rules below; and warn for every drop or repair. Sections where a part is inactive ignore supplied patterns for that part.

## 4. Renderer (`flithack/render.py`)

Deterministic, no network. `render(plan, parts, out_dir) -> list[warning]`.

- **Drums:** grid chars → hits (`x` vel 100, `o` vel 60), repeated per bar of the section. `fill_last_bar` → replace last half-bar with a simple snare/tom 16th run. GM notes, channel 10.
- **Harmony:** chord symbol per bar → sustained voicing, 3–4 voices around C3–C5, nearest-inversion voice leading (keep common tones, move the rest minimally). Parse symbols forgivingly: maj/min/7/m7/sus. If a valid A–G root exists but the quality is unknown, use a plain triad on that root + warning. If even the root is invalid, repeat the previous valid chord; if there is none, use the generated key's tonic triad.
- **Bass / melody:** parse `events_by_bar`, loop each phrase across its section, transpose nothing (the model already wrote real pitches). Octave-shift bass into E1–G3 and melody into C4–C6 + warning, never drop merely for range.
- **Validation (hard rules, from the architecture doc):** snap everything to the 1/16 grid; drop zero/negative durations; clip notes at section boundaries; no overlapping same-pitch notes; drums only on valid GM notes; every file same tempo and length; bar count must match the plan.
- **Forgiving parser rule:** a malformed event token or wrong-length grid row is repaired if obvious (truncate/pad grid to 16, skip one bad token) and logged as a warning — one bad token from the LLM must never kill the run. If an entire part is unusable, render the others and warn `part_dropped:<name>`. Still write that part's valid empty MIDI with tempo and an end-of-track at the shared song length so the required output contract remains intact.
- **Fixed preview programs:** use zero-based GM programs `33` Electric Bass (finger), `48` String Ensemble 1 for harmony, and `80` Lead 1 (square) for melody. These are renderer constants, not user-selectable instrumentation. They keep `song.mid` tracks distinguishable in FluidSynth while staying inside the no-timbre-UI scope.

## 5. UI additions (`app.py`)

Below the analysis results, once `analysis_output/` exists:

1. Text input for the prompt (placeholder: *"calm exploration theme, darker than the reference"*). Empty prompt is allowed → "same vibe as the reference."
2. **Generate** button → runs plan → parts → render → preview with `st.status` per stage. Same rerun rules as SPEC_2 §2: generation runs only on click, results live in `session_state` + on disk.
3. Results: plan summary (key, bpm, section table with chords), download button per `.mid`, `song.mid`, "Download generation_output.zip", and `st.audio` for `preview.wav`.
4. **Regenerate** button → new `<n>` folder, previous completed results stay listed. Add the run number as a variation nonce to the PARTS prompt and provenance so regeneration is explicitly asked for a different valid variation rather than silently reusing a response. Cheap A/B demos well.
5. If `OPENAI_API_KEY` is missing, disable the prompt and Generate/Regenerate controls with the same "no API key" message as SPEC_2. Still list, play, and download previously completed generations.

## 6. Preview (`flithack/preview.py`)

The demo must not end at "here are some MIDI files" — the room has to *hear* the new track.

- **Primary: FluidSynth.** `brew install fluid-synth`, one General MIDI SoundFont (FluidR3_GM.sf2, free) downloaded once to `assets/` (gitignored). Resolve a relative `SOUNDFONT_PATH` from the repository root, verify it exists, then render via subprocess: `fluidsynth -ni <soundfont> song.mid -F preview.wav -r 44100`. Deterministic, exact notes, supports drums.
- Timebox 30 minutes. **Fallback:** skip `preview.wav`, warn `preview_unavailable`, and demo by dragging `song.mid` into GarageBand/Ableton live — which doubles as proof of the "editable in your DAW" pitch, so the fallback is honestly almost as good.
- Preview failure never fails the run; the MIDI package is the product.

## 7. Environment

```bash
brew install fluid-synth
# .env additions (optional):
GENERATION_MODEL=            # empty → use OPENAI_MODEL
GENERATION_TIMEOUT_SECONDS=300
SOUNDFONT_PATH=assets/FluidR3_GM.sf2
```

Resolve the model with `os.getenv("GENERATION_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini")`; an explicitly empty `GENERATION_MODEL=` must never be sent to the API.

Add `generation_output/` and `assets/` to `.gitignore` (`runs/` is already covered).

## 8. Build order and gates

### 1. Renderer from a hardcoded fixture

Write a hand-made `GenerationPlan` + `Parts` fixture (JSON in `tests/fixtures/`) and render it — **no LLM involved**. Add one intentionally messy fixture containing a 15-character drum row, one malformed event, one out-of-range pitch, and one unknown chord quality.

Gate: the four MIDI files + `song.mid` open in a DAW, play together, and sound intentional. The messy fixture also produces all required MIDI, records repairs in `warnings`, and never raises. This is the same "hour 0" gate as the original architecture doc, and it de-risks everything downstream.

### 2. Preview

Gate: one command turns the fixture's `song.mid` into an audible `preview.wav` (or the fallback decision is made and recorded).

### 3. PLAN call

Gate: two different reference profiles + the same prompt produce clearly different plans (key/bpm/chords/energy visibly track each profile). One profile + two different prompts ("darker", "faster combat") produce visibly different plans.

### 4. PARTS call + full CLI

`python -m flithack generate runs/<sha>/analysis_output/ -p "..." -o generation_output/`

Gate: end-to-end run produces a listenable preview, and the new melody/bass are *not* copies of the reference transcription (eyeball the MIDI against `melody.mid` from block A — different notes, similar character).

### 5. UI wiring

Gate: the full demo path — drop mp3 → analysis appears → type prompt → Generate → hear preview → download MIDI → open in DAW and edit a note. That last step *is* the pitch; rehearse it.

### 6. Stretch

- Per-part regenerate ("keep everything, redo the melody").
- A remix-vs-inspired knob (remix: reuse reference chord timeline verbatim; inspired: current behavior).
- Velocity humanization (±10 velocity, ±10ms) in the renderer.
- Second contrasting demo reference through the whole flow.

## 9. Explicitly not doing

- No audio-generation models (Suno-style, MusicGen) — MIDI is the product and the differentiator.
- No instrument/timbre selection beyond the GM SoundFont preview.
- No MIDI keyboard input (the architecture doc's Mode A–D) — post-hackathon.
- No similarity slider, originality scoring, or copyright filter.
- No tempo or meter changes inside a generated track; 4/4 only, one bpm.
- No full-song forms beyond 32 bars; loops are the demo.
- No per-note editing UI — "edit it in your DAW" is the pitch, not a feature to rebuild.
- No streaming generation, no queue; one synchronous run per click.
