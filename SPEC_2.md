# SPEC_2 — UI + LLM Interpretation of the MIDI

Extends `SPEC.md`. Same rules: **functional and dirty**, stable boundaries, elegance inside stages does not matter.

This part adds three things on top of block A:

1. A drag-and-drop web UI that runs the pipeline on an mp3.
2. A MIDI → text layer so an LLM can actually read what we transcribed (drums decoded as kick/snare/hat, not as pitches).
3. An OpenAI (GPT mini) call that interprets that text and writes its interpretation back into `analysis_output/`.

Scope note: `SPEC.md` §10 said "no web UI in this repo." That ban now applies to the **pipeline package** (`flithack/`) only. The UI is a thin separate entry point (`app.py`) that imports the stage functions. The pipeline must keep working headless via the CLI — the UI is a wrapper, never a dependency.

## 1. What this adds to the flow

```text
mp3 (drag & drop)
    ↓
[UI] app.py ── calls ──► flithack pipeline (SPEC.md stages 1–8, unchanged)
    ↓                          ↓
    ↓                    analysis_output/
    ↓                          ↓
    ↓                  [9] midi → text        (midi_repr.py)
    ↓                          ↓
    ↓                 [10] LLM interpretation (interpret.py, OpenAI gpt mini)
    ↓                          ↓
    └──── shows ◄──── llm_interpretation.json (added to analysis_output/)
```

Contract impact: `analysis_output/` gains one file, `llm_interpretation.json`. This is an **additive** change, allowed by SPEC.md §4. The downstream generator can consume it or ignore it.

## 2. UI

**Stack: Streamlit.** One file, `app.py`, repo root, run with `streamlit run app.py`. One local Streamlit process; no separate backend/API service, auth, or deployment. Fallback if Streamlit fights: Gradio. Timebox 20 minutes, same rule as always.

Required elements, top to bottom:

1. `st.file_uploader` accepting `.mp3` / `.wav` (drag & drop is built in), followed by an explicit **Analyze** button. Hash the uploaded bytes and save them under `runs/<sha256-prefix>/`; never share one mutable working directory between uploads. Call the same stage functions the CLI uses — **not** a subprocess of the CLI, so stage caching and error messages come through.
2. Progress: `st.status` block per stage (ingest → timeline → separate → analyze → chords → transcribe → melody/postprocess → package → MIDI representation → interpret). A stage failure shows the exception; it must not die silently.
3. Results, once done:
   - Metrics row: BPM, key, meter, duration.
   - Chord timeline (plain `st.dataframe` is fine).
   - `st.audio` player per stem (drums/bass/vocals/other).
   - Download button per `.mid` file + one "Download analysis_output.zip".
   - Interpretation panel: the fields of `llm_interpretation.json`, rendered as text. If the OpenAI call failed, show the warning and everything else anyway.
4. A **Run LLM interpretation** checkbox. Default on when `OPENAI_API_KEY` exists; otherwise default off and disabled with a "no API key configured" message. Turning it off must skip all network work.

### Streamlit rerun rule

Streamlit reruns `app.py` after widget interactions. The expensive pipeline must run **only** when the Analyze button is clicked:

- Store the current upload hash, completed `analysis_output/` path, and stage/result status in `st.session_state`.
- On later reruns, render the stored results without calling the pipeline again.
- Clicking a MIDI/ZIP download or changing a display control must not repeat model inference.
- A different upload hash creates/selects a different run directory.
- The on-disk completion markers from `SPEC.md` remain the source of truth if the browser session is refreshed.

Not required: piano-roll rendering, waveform plots, session history, styling. Ugly is fine.

## 3. MIDI → text (`flithack/midi_repr.py`)

An LLM cannot read a `.mid` binary — "upload the MIDI to GPT" concretely means: parse with `pretty_midi`, serialize to compact text, put that text in the prompt.

One public function: `midi_repr(analysis_output_dir) -> str`.

### 3.1 Drums are instruments, not notes

In GM percussion, the "pitch" field is an instrument ID. Never show the model raw numbers or note names for drums — decode first:

```python
DRUM_MAP = {36: "KICK", 38: "SNARE", 42: "CLOSED_HAT", 47: "TOM", 49: "CYMBAL"}
```

(Only these five exist in our files, per SPEC.md's drum conventions. If ADTOF emits 35, postprocess already remapped it to 36.)

Serialize drums as **step grids**, one row per instrument, 16 steps per bar, aligned to the detected beat grid — the most LLM-legible rhythm format there is:

```text
bar 5 | KICK   x...x...x...x...
      | SNARE  ....x.......x...
      | HAT    x.x.x.x.x.x.x.x.
```

Use velocity buckets if cheap (`x` accent / `o` soft), skip if not.

The 16-step grid assumes `4/4`, which is the MVP constraint in `SPEC.md`. If the profile is not `4/4`, emit a compact beat-position event list instead and include `midi_repr_non_4_4` in the representation header; never silently force another meter into a 16-step bar.

### 3.2 Pitched tracks (bass, melody, other; omit raw vocals)

Event list per bar, times in beats not seconds, pitch as note name, duration in beats, velocity bucketed:

```text
bass bar 5: F2@1.0 len1.0 | F2@2.5 len0.5 | Ab2@3.0 len1.0 | C3@4.0 len0.5
```

For `other.mid` don't list every note (it's messy by design): per bar, emit the pitch-class set and register span only (`bar 5: {F,Ab,C} span C3-Ab4, 7 notes`).

Do not serialize `vocals.mid` separately. `melody.mid` is already its cleaned melodic representation (or the `other.mid` fallback), so including both wastes tokens and double-counts the same phrase.

### 3.3 Token budget

Do not send the whole song. The payload is:

1. Header: BPM, key, meter, duration, coarse sections, compressed chord summary, energy summary, and per-stem densities from `reference_profile.json`. Do not paste the full per-bar energy and chord arrays.
2. Select representative bars **once for the whole song**: the first 8 bars plus the consecutive 8-bar window with the highest summed energy. Deduplicate overlap, handle songs shorter than 16 bars, and use the same selected bar indices for drums, bass, melody, and other so the model can compare their relationships.
3. Include detailed chords only for those selected bars. Run-length-compress consecutive repeated chords.
4. Target: whole prompt under ~8k tokens. Print `len(text)//4` as a crude token count during development and trim if it explodes.

## 4. LLM interpretation (`flithack/interpret.py`)

**SDK:** `openai` v2.x. **Model:** from env, default `gpt-5-mini`. The client reads `OPENAI_API_KEY` from the environment automatically. Use structured outputs via `responses.parse` + Pydantic — no JSON-in-a-string parsing.

```python
import os
from pydantic import BaseModel
from openai import OpenAI

class DrumInterpretation(BaseModel):
    groove_description: str      # "four-on-the-floor kick, backbeat snare..."
    feel: str                    # straight / swung / shuffled
    density: str                 # sparse / medium / dense
    signature_elements: list[str]

class Interpretation(BaseModel):
    structural_traits: list[str] # "minor-key", "syncopated", "four-on-the-floor"
    overall_character: str
    drums: DrumInterpretation
    bass_behavior: str           # rhythm, contour, relation to kick/roots
    melody_behavior: str         # range, phrasing, contour, repetition
    harmony_color: str           # chord qualities, movement, mood
    energy_arc: str              # how intensity evolves across the song
    generation_hints: list[str]  # concrete advice for composing something inspired by this

client = OpenAI(
    timeout=30.0,
    max_retries=1,  # one SDK-managed retry for transient failures
)

rsp = client.responses.parse(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    input=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": midi_text},
    ],
    text_format=Interpretation,
)
interp = rsp.output_parsed
if interp is None:
    raise RuntimeError("OpenAI returned no parsed interpretation")
```

System prompt requirements: you are a music producer reading a transcription of a reference track; drums lines are instrument step grids, not pitches; describe character and give reusable generation hints; do not reproduce the reference verbatim; be concrete ("syncopated 16th-note hats with accents on the off-beat of 4"), not generic ("energetic drums"). MIDI text contains rhythm, pitch, harmony, density, and velocity but not production timbre: do not invent instruments, sound design, or confident genre labels that are not supported by the supplied representation.

Rules:

- Output written to `analysis_output/llm_interpretation.json`: the Pydantic model dump plus `model`, `schema_version: "0.1"`, `prompt_version`, `prompt_sha256`, and `input_sha256`.
- Use the SDK's configured single retry; do not add a second manual retry loop. On any final API error, refusal, incomplete response, or missing parsed result, append `llm_interpretation_failed` to `warnings` in `reference_profile.json` and continue. **The LLM stage must never fail the pipeline** — it's an enricher, everything upstream is still valid output.
- Update `reference_profile.json` atomically. A later successful interpretation removes a previous `llm_interpretation_failed`; intentionally skipping the LLM neither adds nor removes that warning.
- Also expose it on the CLI: `python -m flithack interpret analysis_output/` — so it can rerun on cached analysis without re-separating anything. `--force` always calls the API again.
- Interpretation caching is valid only when `model`, `prompt_sha256`, and `input_sha256` all match. The prompt hash covers the complete system prompt and schema instructions; the input hash covers the exact `midi_repr()` text. Prompt iteration must never silently reuse stale JSON.
- Cost sanity: one call per song, mini-tier model, <10k tokens in / ~1k out — pennies. No cost engineering needed.

## 5. Environment

```bash
pip install streamlit openai python-dotenv
```

After the dependency spike works, record and pin the exact installed versions used by the demo. Do not leave a successful hackathon environment floating on unconstrained latest packages.

`.env` at repo root, **gitignored**, loaded with `load_dotenv()` at the top of `cli.py` and `app.py`:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini
```

Commit `.env.example` with the same keys and a placeholder value. Add to `.gitignore`: `.env`, `analysis_output/`, `*.wav` working files.

If `OPENAI_API_KEY` is missing: the interpret stage reports "no API key configured" and skips; the UI disables **Run LLM interpretation** — never a crash, never a hang.

## 6. Build order and gates

### 1. `midi_repr.py` against an existing `analysis_output/`

Gate: printing the text for the demo song, a human can read the drum grid and hum the bassline. If a human can't read it, the model can't either.

### 2. `interpret.py` on the CLI

Gate: `python -m flithack interpret analysis_output/` writes a valid `llm_interpretation.json`, and the `generation_hints` mention something actually true about the song (e.g. it notices the four-on-the-floor kick). If the output is generic slop, fix the prompt/representation, not the schema.

### 3. UI thin slice

Gate: drag an mp3, click Analyze, watch stages progress, and download the zip. Click every download button and change a display control; the pipeline must not run again. No LLM yet.

### 4. Wire interpretation into the UI

Gate: full flow — drop mp3, see profile + stems + MIDI downloads + interpretation panel. Kill the Wi-Fi, run again with **Run LLM interpretation** off: still works end to end. With it on, network failure stops after the configured timeout/retry and still shows all upstream results.

### 5. Stretch

- Velocity accents in drum grids.
- Show the drum grid text in the UI (it demos surprisingly well).
- Second demo song through the full UI flow.

## 7. Explicitly not doing

- No deployment, auth, DB, persisted session history, separate backend, or job queue — localhost Streamlit only. In-memory `st.session_state` is allowed for rerun safety.
- No prompt interpretation for *generation* and no new-music calls — that is still block B (teammate). This LLM call only *describes* the reference.
- No fine-tuning, no embeddings, no RAG. One prompt, one structured response.
- No audio upload to OpenAI — text only. (Audio-language analysis of the raw mix stays a stretch idea from the original architecture doc, not this weekend.)
- No streaming token UI, no cost dashboards.
- No React/Next frontend. Streamlit is the UI.
