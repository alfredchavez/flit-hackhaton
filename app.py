"""Streamlit UI wrapper — calls flithack stage functions (not CLI subprocess)."""

from __future__ import annotations

import hashlib
import io
import json
import traceback
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from flithack.cli import ingest
from flithack.generate import generate_track, list_completed_generations
from flithack.interpret import (
    has_api_key,
    interpret_analysis,
    load_valid_cached_interpretation,
)
from flithack.midi_repr import midi_repr

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --fh-panel: #211d17;
  --fh-line: #332d24;
  --fh-text: #e6e1d8;
  --fh-dim: #a1978a;
  --fh-faint: #6f675c;
  --fh-accent: #e8963c;
  --fh-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --fh-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: var(--fh-sans);
}

/* Chrome */
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.block-container {
  padding-top: 2.25rem;
  padding-bottom: 4rem;
  max-width: 1180px;
}

/* Wordmark */
.fh-wordmark {
  font-family: var(--fh-mono);
  font-weight: 700;
  font-size: 26px;
  letter-spacing: -0.02em;
  color: var(--fh-text);
  line-height: 1.1;
}
.fh-tagline {
  font-family: var(--fh-mono);
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fh-dim);
  margin: 4px 0 4px;
}

/* Section labels */
.fh-label {
  font-family: var(--fh-mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fh-faint);
  margin: 20px 0 4px;
}

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--fh-line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.75rem; }

/* Metrics: studio readouts */
[data-testid="stMetric"] {
  background: var(--fh-panel);
  border: 1px solid var(--fh-line);
  border-radius: 6px;
  padding: 12px 14px 10px;
}
[data-testid="stMetricLabel"] p {
  font-family: var(--fh-mono);
  font-size: 10.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fh-faint);
}
[data-testid="stMetricValue"] {
  font-family: var(--fh-mono);
  font-size: 26px;
  color: var(--fh-text);
}

/* Tabs */
button[data-baseweb="tab"] {
  font-family: var(--fh-mono);
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--fh-accent); }

/* Data, code */
code, pre, [data-testid="stCodeBlock"] { font-family: var(--fh-mono) !important; }

/* Captions */
[data-testid="stCaptionContainer"] {
  font-family: var(--fh-mono);
  font-size: 11px;
  color: var(--fh-dim);
}

/* Buttons */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
  font-family: var(--fh-mono);
  font-size: 12.5px;
}

/* Expanders & pipeline status */
[data-testid="stExpander"] {
  border: 1px solid var(--fh-line);
  border-radius: 6px;
  background: var(--fh-panel);
}

/* Chips */
.fh-chip {
  display: inline-block;
  font-family: var(--fh-mono);
  font-size: 11px;
  color: var(--fh-text);
  border: 1px solid var(--fh-line);
  border-radius: 999px;
  padding: 2px 10px;
  margin: 0 6px 6px 0;
  background: var(--fh-panel);
}

/* Empty state */
.fh-empty {
  border: 1px solid var(--fh-line);
  border-radius: 8px;
  background: var(--fh-panel);
  padding: 20px 24px;
  margin-top: 20px;
  max-width: 620px;
}
.fh-step { display: flex; gap: 14px; padding: 10px 0; align-items: baseline; }
.fh-step + .fh-step { border-top: 1px solid var(--fh-line); }
.fh-num {
  font-family: var(--fh-mono);
  color: var(--fh-accent);
  font-weight: 600;
  font-size: 13px;
  min-width: 22px;
}
.fh-step p { margin: 0; font-size: 14px; color: var(--fh-text); }
.fh-step small { display: block; color: var(--fh-dim); font-size: 12px; margin-top: 2px; }

/* Selection & focus */
::selection { background: rgba(232, 150, 60, 0.28); }
*:focus-visible { outline: 2px solid var(--fh-accent) !important; outline-offset: 1px; }

/* Audio full width */
[data-testid="stAudio"] { width: 100%; }

hr { border-color: var(--fh-line); }
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _label(text: str) -> None:
    st.markdown(f'<div class="fh-label">{text}</div>', unsafe_allow_html=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_dir(sha: str) -> Path:
    return RUNS_DIR / sha[:16]


def _zip_dir(output_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(output_dir)
            if ".work" in rel.parts or any(part.endswith(".partial") for part in rel.parts):
                continue
            if rel.parts and rel.parts[0].startswith("."):
                continue
            zf.write(path, arcname=str(rel))
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _midi_text_for(output_dir_str: str, profile_mtime: float) -> str | None:
    """midi_repr() for a run, cached on the profile's mtime so reruns are free."""
    try:
        return midi_repr(Path(output_dir_str))
    except Exception:  # noqa: BLE001 — the expander simply stays hidden
        return None


def _discover_runs() -> list[tuple[str, Path]]:
    """(label, analysis_output path) for every completed run on disk, newest first."""
    found = []
    if not RUNS_DIR.is_dir():
        return found
    for d in RUNS_DIR.iterdir():
        profile = d / "analysis_output" / "reference_profile.json"
        if not profile.is_file():
            continue
        try:
            source = json.loads(profile.read_text()).get("source_file", "?")
        except (json.JSONDecodeError, OSError):
            continue
        found.append((profile.stat().st_mtime, f"{source} · {d.name[:8]}", d / "analysis_output"))
    found.sort(key=lambda t: t[0], reverse=True)
    return [(label, path) for _, label, path in found]


def _load_run(output_dir: Path, source_label: str) -> None:
    """Point the session at a completed run restored from disk."""
    st.session_state.output_dir = str(output_dir)
    st.session_state.upload_hash = None
    st.session_state.source_name = source_label
    st.session_state.interp = None
    st.session_state.result_ok = True
    st.session_state.result_error = None


def _run_pipeline(
    input_path: Path,
    output_dir: Path,
    work_dir: Path,
    *,
    run_llm: bool,
    force: bool = False,
    status_container,
    source_name: str | None = None,
) -> dict:
    """Execute stages 1–10 with per-stage st.status updates. Returns result dict."""
    from flithack.analyze import analyze_global, compute_per_stem_stats
    from flithack.chords import estimate_chords
    from flithack.package import build_profile, package_output
    from flithack.postprocess import build_melody, postprocess_part
    from flithack.separate import separate_stems
    from flithack.timeline import align_timeline
    from flithack.transcribe import transcribe_stem

    quantize = True
    result: dict = {"ok": False, "error": None, "output_dir": str(output_dir)}

    try:
        with status_container.status("1 · Ingest", expanded=True) as s:
            normalized = work_dir / "normalized.wav"
            if force or not normalized.is_file():
                s.write(f"ffmpeg → {normalized.name}")
                ingest(input_path, normalized)
            else:
                s.write("cached")
            s.update(label="1 · Ingest ✓", state="complete", expanded=False)

        with status_container.status("2 · Timeline alignment", expanded=True) as s:
            timeline = align_timeline(normalized, work_dir / "timeline", force=force)
            s.write(
                f"offset={timeline.source_offset_seconds:.3f}s "
                f"bpm={timeline.bpm:.2f} origin={timeline.timeline_origin}"
            )
            s.update(label="2 · Timeline alignment ✓", state="complete", expanded=False)

        with status_container.status("3 · Stem separation", expanded=True) as s:
            stem_paths = separate_stems(
                timeline.aligned_wav, work_dir / "stems", force=force
            )
            s.write(", ".join(stem_paths.keys()))
            s.update(label="3 · Stem separation ✓", state="complete", expanded=False)

        with status_container.status("4 · Global analysis", expanded=True) as s:
            analysis = analyze_global(
                timeline.aligned_wav,
                beats=timeline.beats,
                downbeats=timeline.downbeats,
                bpm=timeline.bpm,
                meter=timeline.meter,
                duration_seconds=timeline.duration_seconds,
                extra_warnings=list(timeline.warnings),
            )
            s.write(f"key={analysis.key} meter={analysis.meter}")
            s.update(label="4 · Global analysis ✓", state="complete", expanded=False)

        with status_container.status("5 · Chords", expanded=True) as s:
            chords, chord_warnings = estimate_chords(
                timeline.aligned_wav,
                analysis.beats,
                duration_seconds=analysis.duration_seconds,
            )
            all_warnings = list(analysis.warnings) + list(chord_warnings)
            s.write(f"{len(chords)} chord segments")
            s.update(label="5 · Chords ✓", state="complete", expanded=False)

        with status_container.status("6 · Transcribe stems", expanded=True) as s:
            raw_midi_dir = work_dir / "midi_raw"
            raw_midi_dir.mkdir(parents=True, exist_ok=True)
            raw_paths = {}
            for stem_type in ("drums", "bass", "vocals", "other"):
                s.write(f"transcribing {stem_type}…")
                raw_paths[stem_type] = transcribe_stem(
                    stem_paths[stem_type],
                    stem_type,
                    raw_midi_dir / f"{stem_type}.mid",
                    bpm=analysis.bpm,
                    force=force,
                )
            s.update(label="6 · Transcribe stems ✓", state="complete", expanded=False)

        with status_container.status("7 · Melody / postprocess", expanded=True) as s:
            final_midi_dir = work_dir / "midi_final"
            final_midi_dir.mkdir(parents=True, exist_ok=True)
            midi_paths = {}
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
            per_stem = compute_per_stem_stats(
                midi_paths, analysis.downbeats, analysis.duration_seconds
            )
            s.update(label="7 · Melody / postprocess ✓", state="complete", expanded=False)

        with status_container.status("8 · Package", expanded=True) as s:
            profile = build_profile(
                source_file=source_name or input_path.name,
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
            s.write(str(output_dir))
            s.update(label="8 · Package ✓", state="complete", expanded=False)

        midi_text = None
        midi_repr_error = None
        with status_container.status("9 · MIDI representation", expanded=True) as s:
            try:
                midi_text = midi_repr(output_dir)
                s.write(f"~{len(midi_text) // 4} tokens")
                s.update(label="9 · MIDI representation ✓", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001 — Block A remains usable
                midi_repr_error = str(exc)
                (output_dir / "llm_interpretation.json").unlink(missing_ok=True)
                s.write(f"unavailable: {exc}")
                s.update(
                    label="9 · MIDI representation failed (analysis still valid)",
                    state="error",
                )

        interp_result = None
        with status_container.status("10 · LLM interpretation", expanded=True) as s:
            if midi_text is None:
                interp_result = {
                    "ok": False,
                    "skipped": True,
                    "error": f"midi representation unavailable: {midi_repr_error}",
                    "interpretation": None,
                }
                s.write(interp_result["error"])
                s.update(label="10 · LLM interpretation skipped", state="complete")
            elif run_llm:
                interp_result = interpret_analysis(
                    output_dir,
                    force=force,
                    skip_network=False,
                    midi_text=midi_text,
                )
                if interp_result.get("ok"):
                    s.write("ok" + (" (cached)" if interp_result.get("cached") else ""))
                    s.update(
                        label="10 · LLM interpretation ✓",
                        state="complete",
                        expanded=False,
                    )
                elif interp_result.get("skipped"):
                    s.write(interp_result.get("error") or "skipped")
                    s.update(label="10 · LLM interpretation skipped", state="complete")
                else:
                    s.write(f"failed: {interp_result.get('error')}")
                    s.update(
                        label="10 · LLM interpretation failed (analysis still valid)",
                        state="error",
                    )
            else:
                interp_result = interpret_analysis(
                    output_dir,
                    force=False,
                    skip_network=True,
                    midi_text=midi_text,
                )
                if interp_result.get("cached"):
                    s.write("cached (no network call)")
                    s.update(
                        label="10 · LLM interpretation ✓ (cached)",
                        state="complete",
                        expanded=False,
                    )
                else:
                    s.write("skipped (checkbox off / no key)")
                    s.update(label="10 · LLM interpretation skipped", state="complete")

        result["ok"] = True
        result["interp"] = interp_result
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{exc}\n{traceback.format_exc()}"
        return result


def _render_interpretation(data: dict) -> None:
    overall = data.get("overall_character")
    if overall:
        st.markdown(overall)

    traits = data.get("structural_traits") or []
    if traits:
        st.markdown(
            "".join(f'<span class="fh-chip">{t}</span>' for t in traits),
            unsafe_allow_html=True,
        )

    drums = data.get("drums") or {}
    col_drums, col_parts = st.columns(2)
    with col_drums:
        _label("Drums")
        for key, name in (
            ("groove_description", "Groove"),
            ("feel", "Feel"),
            ("density", "Density"),
            ("signature_elements", "Signature"),
        ):
            value = drums.get(key)
            if value:
                st.markdown(f"**{name}:** {value}")
    with col_parts:
        _label("Parts")
        for key, name in (
            ("bass_behavior", "Bass"),
            ("melody_behavior", "Melody"),
            ("harmony_color", "Harmony"),
            ("energy_arc", "Energy arc"),
        ):
            value = data.get(key)
            if value:
                st.markdown(f"**{name}:** {value}")

    hints = data.get("generation_hints") or []
    if hints:
        _label("Generation hints")
        for hint in hints:
            st.markdown(f"- {hint}")

    st.caption(
        f"model {data.get('model')} · schema {data.get('schema_version')} "
        f"· prompt_v {data.get('prompt_version')}"
    )


def _render_overview_tab(
    output_dir: Path, profile: dict, midi_text: str | None, interp: dict | None
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BPM", f"{profile.get('bpm', 0):.1f}")
    c2.metric("Key", str(profile.get("key", "?")))
    c3.metric("Meter", str(profile.get("meter", "?")))
    c4.metric("Duration (s)", f"{profile.get('duration_seconds', 0):.1f}")

    if profile.get("warnings"):
        with st.expander(f"Warnings ({len(profile['warnings'])})", expanded=False):
            for w in profile["warnings"]:
                st.markdown(f"- `{w}`")

    energy = profile.get("energy_curve") or []
    if energy:
        _label("Energy per bar")
        st.area_chart(
            {"energy": [e.get("value", 0.0) for e in energy]},
            height=160,
            color="#e8963c",
        )

    sections = profile.get("sections") or []
    if sections:
        _label("Sections")
        st.dataframe(sections, width="stretch", hide_index=True)

    _label("Chords")
    chords = profile.get("chords") or []
    if chords:
        rows = [
            {
                "chord": c.get("chord"),
                "start": round(float(c.get("start", 0.0)), 2),
                "end": round(float(c.get("end", 0.0)), 2),
            }
            for c in chords
        ]
        st.dataframe(rows, width="stretch", height=240, hide_index=True)
    else:
        st.info("No chords (or chords_unreliable)")

    _label("LLM interpretation")
    data = None
    if interp and interp.get("interpretation"):
        data = interp["interpretation"]
    else:
        # Fresh Streamlit session / loaded run: restore only a provenance-valid cache.
        data = load_valid_cached_interpretation(output_dir, midi_text=midi_text)

    if data:
        _render_interpretation(data)
    else:
        if interp and interp.get("error"):
            st.warning(f"Interpretation unavailable: {interp['error']}")
        elif "llm_interpretation_failed" in (profile.get("warnings") or []):
            st.warning("llm_interpretation_failed (see profile warnings)")
        else:
            st.info("No interpretation yet (skipped or not run).")


def _render_stems_tab(output_dir: Path) -> None:
    stems_dir = output_dir / "stems"
    names = [n for n in ("drums", "bass", "vocals", "other") if (stems_dir / f"{n}.wav").is_file()]
    if not names:
        st.info("No stems found in output.")
        return
    for row_start in range(0, len(names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, names[row_start : row_start + 2]):
            with col:
                _label(name)
                st.audio(str(stems_dir / f"{name}.wav"))


def _render_midi_tab(output_dir: Path, midi_text: str | None) -> None:
    _label("Parts")
    midi_dir = output_dir / "midi"
    cols = st.columns(5)
    for i, name in enumerate(("drums", "bass", "vocals", "other", "melody")):
        p = midi_dir / f"{name}.mid"
        if p.is_file():
            cols[i].download_button(
                label=f"{name}.mid",
                data=p.read_bytes(),
                file_name=f"{name}.mid",
                mime="audio/midi",
                key=f"dl_midi_{name}_{output_dir.name}",
                width="stretch",
            )

    st.download_button(
        label="Download analysis_output.zip",
        data=_zip_dir(output_dir),
        file_name="analysis_output.zip",
        mime="application/zip",
        key=f"dl_zip_{output_dir.name}",
    )

    if midi_text:
        with st.expander("MIDI text representation (what the LLM sees)", expanded=False):
            st.code(midi_text, language=None)


def _render_results(output_dir: Path, interp: dict | None) -> None:
    profile_path = output_dir / "reference_profile.json"
    if not profile_path.is_file():
        st.error("No reference_profile.json in output")
        return
    profile = json.loads(profile_path.read_text())
    # Recomputed (cached on mtime) so restored sessions get the expander + interp too.
    midi_text = _midi_text_for(str(output_dir), profile_path.stat().st_mtime)

    st.caption(f"{profile.get('source_file', '?')} · {output_dir}")

    tab_overview, tab_stems, tab_midi = st.tabs(["Overview", "Stems", "MIDI"])
    with tab_overview:
        _render_overview_tab(output_dir, profile, midi_text, interp)
    with tab_stems:
        _render_stems_tab(output_dir)
    with tab_midi:
        _render_midi_tab(output_dir, midi_text)

    st.divider()
    st.markdown(
        '<div class="fh-wordmark" style="font-size:19px">Generate a new track</div>',
        unsafe_allow_html=True,
    )
    _render_generation_panel(output_dir)


def _render_one_generation(gen_dir: Path) -> None:
    plan_path = gen_dir / "generation_plan.json"
    plan_doc = {}
    if plan_path.is_file():
        try:
            plan_doc = json.loads(plan_path.read_text())
        except json.JSONDecodeError:
            plan_doc = {}
    plan = plan_doc.get("plan") or {}
    st.markdown(f"**{plan.get('title', gen_dir.name)}**")
    st.caption(
        f"bpm {plan.get('bpm', '?')} · key {plan.get('key', '?')} "
        f"· meter {plan.get('meter', '?')}"
    )
    if plan_doc.get("user_prompt"):
        st.caption(f"prompt: {plan_doc['user_prompt']}")

    preview = gen_dir / "preview.wav"
    if preview.is_file():
        _label("Preview")
        st.audio(str(preview))
    else:
        st.info("No preview.wav (preview_unavailable); open song.mid in a DAW.")

    sections = plan.get("sections") or []
    if sections:
        rows = [
            {
                "section": s.get("name") or s.get("id"),
                "bars": s.get("bars"),
                "energy": s.get("energy"),
                "parts": ", ".join(s.get("active_parts") or []),
                "chords": " → ".join(s.get("chords") or []),
            }
            for s in sections
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

    warns = plan_doc.get("warnings") or []
    if warns:
        with st.expander(f"Warnings ({len(warns)})", expanded=False):
            for w in warns:
                st.markdown(f"- `{w}`")

    _label("Downloads")
    midi_dir = gen_dir / "midi"
    cols = st.columns(5)
    for i, name in enumerate(("drums", "bass", "harmony", "melody", "song")):
        p = gen_dir / "song.mid" if name == "song" else midi_dir / f"{name}.mid"
        if p.is_file():
            cols[i].download_button(
                label=f"{name}.mid",
                data=p.read_bytes(),
                file_name=f"{gen_dir.name}_{name}.mid",
                mime="audio/midi",
                key=f"dl_gen_{gen_dir.name}_{name}",
                width="stretch",
            )

    st.download_button(
        label="Download generation_output.zip",
        data=_zip_dir(gen_dir),
        file_name=f"generation_{gen_dir.name}.zip",
        mime="application/zip",
        key=f"dl_gen_zip_{gen_dir.name}",
    )


def _render_generation_panel(analysis_output_dir: Path) -> None:
    st.caption(
        "Block B: plan → parts → render → preview. "
        "Uses only analysis_output/ + your prompt; never re-reads the reference audio."
    )

    api_ok = has_api_key()
    gen_root = analysis_output_dir.parent / "generation_output"
    select_key = f"gen_select_{gen_root}"

    prompt = st.text_input(
        "Generation prompt",
        placeholder="calm exploration theme, darker than the reference",
        disabled=not api_ok,
        key="gen_prompt",
    )

    gen_clicked = st.button(
        "Generate new track",
        type="primary",
        disabled=not api_ok,
        key="btn_generate",
    )
    st.caption("Each click creates a new numbered variation; previous ones stay listed.")
    if not api_ok:
        st.info("No API key configured; generation disabled (existing results still listed).")

    if gen_clicked:
        try:
            with st.status(
                "11–14 · Plan → Parts → Render → Preview", expanded=True
            ) as s:
                s.write(f"prompt: {prompt or 'same vibe as the reference'}")
                result = generate_track(
                    analysis_output_dir,
                    user_prompt=prompt or "",
                    output_dir=gen_root,
                    skip_preview=False,
                )
                s.write(f"→ {result.get('output_dir')}")
                s.update(label="11–14 · Generation ✓", state="complete", expanded=False)
            st.session_state.last_generation = result.get("output_dir")
            st.session_state.gen_error = None
            # Reset the selector so the newest variation is auto-selected below.
            st.session_state.pop(select_key, None)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            st.session_state.gen_error = f"{exc}\n{traceback.format_exc()}"

    if st.session_state.get("gen_error"):
        st.error("Generation failed (analysis + prior generations still OK)")
        with st.expander("Error detail", expanded=False):
            st.code(st.session_state.gen_error)

    completed = list_completed_generations(gen_root)
    if not completed:
        st.write("No completed generations yet.")
        return

    _label("Completed generations")
    labels = [p.name for p in completed]
    default_ix = len(labels) - 1
    last = st.session_state.get("last_generation")
    if last and Path(last).name in labels:
        default_ix = labels.index(Path(last).name)

    choice = st.selectbox(
        "Completed generations",
        options=labels,
        index=default_ix,
        key=select_key,
        label_visibility="collapsed",
    )
    _render_one_generation(gen_root / choice)


def _render_empty_state(uploaded) -> None:
    if uploaded is not None:
        st.info("New audio selected. Click **Analyze** in the sidebar to run the pipeline.")
        return
    st.markdown(
        """
<div class="fh-empty">
  <div class="fh-step">
    <div class="fh-num">01</div>
    <p>Drop a reference track<small>mp3 or wav, in the sidebar on the left</small></p>
  </div>
  <div class="fh-step">
    <div class="fh-num">02</div>
    <p>Analyze<small>ten stages: ingest, timeline, stems, analysis, chords, transcription, postprocess, package, MIDI text, LLM read</small></p>
  </div>
  <div class="fh-step">
    <div class="fh-num">03</div>
    <p>Inspect and generate<small>metrics, stems and MIDI one click away; new tracks from a prompt in the Generate tab</small></p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="OSTify", layout="wide", initial_sidebar_state="expanded"
    )
    _inject_css()

    api_ok = has_api_key()
    if "run_llm" not in st.session_state:
        st.session_state.run_llm = api_ok

    with st.sidebar:
        st.markdown('<div class="fh-wordmark">OSTify</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fh-tagline">reference → profile → new MIDI</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        _label("Input")
        uploaded = st.file_uploader(
            "Reference audio (mp3 / wav)",
            type=["mp3", "wav"],
            accept_multiple_files=False,
        )

        _label("Options")
        run_llm = st.checkbox(
            "LLM interpretation",
            value=st.session_state.run_llm if api_ok else False,
            disabled=not api_ok,
            help="Requires OPENAI_API_KEY in environment / .env",
            key="run_llm_checkbox",
        )
        st.session_state.run_llm = run_llm if api_ok else False
        force = st.checkbox("Force recompute (ignore stage caches)", value=False)

        analyze = st.button(
            "Analyze",
            type="primary",
            disabled=uploaded is None,
            width="stretch",
            key="btn_analyze",
        )

        previous = _discover_runs()
        if previous:
            _label("Previous runs")
            prev_labels = [label for label, _ in previous]
            prev_choice = st.selectbox(
                "Previous runs",
                options=prev_labels,
                key="prev_run_select",
                label_visibility="collapsed",
            )
            if st.button("Load selected run", width="stretch", key="btn_load_run"):
                chosen = dict(previous)[prev_choice]
                _load_run(chosen, prev_choice)

        st.divider()
        if api_ok:
            st.caption("OPENAI_API_KEY: configured")
        else:
            st.caption("OPENAI_API_KEY: missing · LLM steps disabled")

    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    current_upload_sha = (
        _sha256_bytes(uploaded_bytes) if uploaded_bytes is not None else None
    )

    st.markdown('<div class="fh-wordmark">OSTify</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fh-tagline">reference audio → musical profile → new MIDI</div>',
        unsafe_allow_html=True,
    )

    if analyze and uploaded is not None:
        raw = uploaded_bytes if uploaded_bytes is not None else b""
        sha = current_upload_sha or _sha256_bytes(raw)
        rdir = _run_dir(sha)
        rdir.mkdir(parents=True, exist_ok=True)
        ext = Path(uploaded.name).suffix.lower() or ".wav"
        input_path = rdir / f"input{ext}"
        input_path.write_bytes(raw)
        output_dir = rdir / "analysis_output"
        work_dir = rdir / "work"
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        st.session_state.upload_hash = sha
        st.session_state.output_dir = str(output_dir)
        st.session_state.source_name = uploaded.name

        _label("Pipeline")
        progress = st.container()
        result = _run_pipeline(
            input_path,
            output_dir,
            work_dir,
            run_llm=bool(st.session_state.run_llm),
            force=force,
            status_container=progress,
            source_name=uploaded.name,
        )
        st.session_state.result_ok = result.get("ok", False)
        st.session_state.result_error = result.get("error")
        st.session_state.interp = result.get("interp")
        # New analysis output → any mtime-cached midi_repr text is stale.
        _midi_text_for.clear()

        if not result.get("ok"):
            st.error("Pipeline failed")
            with st.expander("Error detail", expanded=True):
                st.code(result.get("error") or "unknown error")
        else:
            st.success("Analysis complete.")

    # Render stored results on reruns (downloads / checkbox changes must not re-run)
    out = st.session_state.get("output_dir")
    stored_matches_upload = (
        current_upload_sha is None
        or current_upload_sha == st.session_state.get("upload_hash")
    )
    if (
        out
        and stored_matches_upload
        and Path(out).is_dir()
        and (Path(out) / "reference_profile.json").is_file()
    ):
        if st.session_state.get("result_error") and not st.session_state.get("result_ok"):
            st.error("Last run failed")
            with st.expander("Error detail", expanded=False):
                st.code(st.session_state.get("result_error"))
        _render_results(Path(out), st.session_state.get("interp"))
    else:
        _render_empty_state(uploaded)


if __name__ == "__main__":
    main()
