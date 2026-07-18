"""Streamlit UI wrapper — calls flithack stage functions (not CLI subprocess)."""

from __future__ import annotations

import hashlib
import io
import json
import os
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


def _zip_analysis_output(output_dir: Path) -> bytes:
    return _zip_dir(output_dir)


def _run_pipeline(
    input_path: Path,
    output_dir: Path,
    work_dir: Path,
    *,
    run_llm: bool,
    force: bool = False,
    status_container,
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
            s.update(label="1 · Ingest ✓", state="complete")

        with status_container.status("2 · Timeline alignment", expanded=True) as s:
            timeline = align_timeline(normalized, work_dir / "timeline", force=force)
            s.write(
                f"offset={timeline.source_offset_seconds:.3f}s "
                f"bpm={timeline.bpm:.2f} origin={timeline.timeline_origin}"
            )
            s.update(label="2 · Timeline alignment ✓", state="complete")

        with status_container.status("3 · Stem separation", expanded=True) as s:
            stem_paths = separate_stems(
                timeline.aligned_wav, work_dir / "stems", force=force
            )
            s.write(", ".join(stem_paths.keys()))
            s.update(label="3 · Stem separation ✓", state="complete")

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
            s.update(label="4 · Global analysis ✓", state="complete")

        with status_container.status("5 · Chords", expanded=True) as s:
            chords, chord_warnings = estimate_chords(
                timeline.aligned_wav,
                analysis.beats,
                duration_seconds=analysis.duration_seconds,
            )
            all_warnings = list(analysis.warnings) + list(chord_warnings)
            s.write(f"{len(chords)} chord segments")
            s.update(label="5 · Chords ✓", state="complete")

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
            s.update(label="6 · Transcribe stems ✓", state="complete")

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
            s.update(label="7 · Melody / postprocess ✓", state="complete")

        with status_container.status("8 · Package", expanded=True) as s:
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
            s.write(str(output_dir))
            s.update(label="8 · Package ✓", state="complete")

        midi_text = None
        midi_repr_error = None
        with status_container.status("9 · MIDI representation", expanded=True) as s:
            try:
                midi_text = midi_repr(output_dir)
                s.write(f"~{len(midi_text) // 4} tokens")
                s.update(label="9 · MIDI representation ✓", state="complete")
            except Exception as exc:  # noqa: BLE001 — Block A remains usable
                midi_repr_error = str(exc)
                (output_dir / "llm_interpretation.json").unlink(missing_ok=True)
                s.write(f"unavailable: {exc}")
                s.update(label="9 · MIDI representation unavailable", state="error")

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
            elif run_llm:
                interp_result = interpret_analysis(
                    output_dir,
                    force=force,
                    skip_network=False,
                    midi_text=midi_text,
                )
                if interp_result.get("ok"):
                    s.write("ok" + (" (cached)" if interp_result.get("cached") else ""))
                elif interp_result.get("skipped"):
                    s.write(interp_result.get("error") or "skipped")
                else:
                    s.write(f"failed: {interp_result.get('error')}")
            else:
                interp_result = interpret_analysis(
                    output_dir,
                    force=False,
                    skip_network=True,
                    midi_text=midi_text,
                )
                if interp_result.get("cached"):
                    s.write("cached (no network call)")
                else:
                    s.write("skipped (checkbox off / no key)")
            s.update(label="10 · LLM interpretation ✓", state="complete")

        result["ok"] = True
        result["profile"] = json.loads(
            (output_dir / "reference_profile.json").read_text()
        )
        result["midi_text"] = midi_text
        result["interp"] = interp_result
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{exc}\n{traceback.format_exc()}"
        return result


def _render_results(output_dir: Path, midi_text: str | None, interp: dict | None) -> None:
    profile_path = output_dir / "reference_profile.json"
    if not profile_path.is_file():
        st.error("No reference_profile.json in output")
        return
    profile = json.loads(profile_path.read_text())

    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BPM", f"{profile.get('bpm', 0):.1f}")
    c2.metric("Key", str(profile.get("key", "?")))
    c3.metric("Meter", str(profile.get("meter", "?")))
    c4.metric("Duration (s)", f"{profile.get('duration_seconds', 0):.1f}")

    if profile.get("warnings"):
        st.warning("warnings: " + ", ".join(profile["warnings"]))

    st.subheader("Chords")
    chords = profile.get("chords") or []
    if chords:
        st.dataframe(chords, use_container_width=True)
    else:
        st.info("No chords (or chords_unreliable)")

    st.subheader("Stems")
    stems_dir = output_dir / "stems"
    for name in ("drums", "bass", "vocals", "other"):
        p = stems_dir / f"{name}.wav"
        if p.is_file():
            st.caption(name)
            st.audio(str(p))

    st.subheader("MIDI downloads")
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
            )

    zip_bytes = _zip_analysis_output(output_dir)
    st.download_button(
        label="Download analysis_output.zip",
        data=zip_bytes,
        file_name="analysis_output.zip",
        mime="application/zip",
        key=f"dl_zip_{output_dir.name}",
    )

    # Stretch: show drum grid / midi text
    if midi_text:
        with st.expander("MIDI text representation (what the LLM sees)", expanded=False):
            st.code(midi_text, language=None)

    st.subheader("LLM interpretation")
    data = None
    if interp and interp.get("interpretation"):
        data = interp["interpretation"]
    elif interp is None:
        # Fresh Streamlit session: restore only a provenance-valid disk cache.
        data = load_valid_cached_interpretation(output_dir, midi_text=midi_text)

    if data:
        st.markdown(f"**Overall:** {data.get('overall_character', '')}")
        st.markdown("**Structural traits:** " + ", ".join(data.get("structural_traits") or []))
        drums = data.get("drums") or {}
        st.markdown("**Drums**")
        st.write(
            {
                "groove": drums.get("groove_description"),
                "feel": drums.get("feel"),
                "density": drums.get("density"),
                "signature": drums.get("signature_elements"),
            }
        )
        st.markdown(f"**Bass:** {data.get('bass_behavior', '')}")
        st.markdown(f"**Melody:** {data.get('melody_behavior', '')}")
        st.markdown(f"**Harmony:** {data.get('harmony_color', '')}")
        st.markdown(f"**Energy arc:** {data.get('energy_arc', '')}")
        st.markdown("**Generation hints**")
        for h in data.get("generation_hints") or []:
            st.markdown(f"- {h}")
        st.caption(
            f"model={data.get('model')} schema={data.get('schema_version')} "
            f"prompt_v={data.get('prompt_version')}"
        )
    else:
        if interp and interp.get("error"):
            st.warning(f"Interpretation unavailable: {interp['error']}")
        elif "llm_interpretation_failed" in (profile.get("warnings") or []):
            st.warning("llm_interpretation_failed (see profile warnings)")
        else:
            st.info("No interpretation yet (skipped or not run).")

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
    st.markdown(
        f"**{plan.get('title', gen_dir.name)}** — "
        f"bpm={plan.get('bpm', '?')} key={plan.get('key', '?')} "
        f"meter={plan.get('meter', '?')}"
    )
    if plan_doc.get("user_prompt"):
        st.caption(f"prompt: {plan_doc['user_prompt']}")
    sections = plan.get("sections") or []
    if sections:
        rows = [
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "bars": s.get("bars"),
                "energy": s.get("energy"),
                "parts": ", ".join(s.get("active_parts") or []),
                "chords": " → ".join(s.get("chords") or []),
            }
            for s in sections
        ]
        st.dataframe(rows, use_container_width=True)

    warns = plan_doc.get("warnings") or []
    if warns:
        st.caption("warnings: " + ", ".join(warns[:12]) + ("…" if len(warns) > 12 else ""))

    midi_dir = gen_dir / "midi"
    cols = st.columns(5)
    for i, name in enumerate(("drums", "bass", "harmony", "melody", "song")):
        if name == "song":
            p = gen_dir / "song.mid"
        else:
            p = midi_dir / f"{name}.mid"
        if p.is_file():
            cols[i].download_button(
                label=f"{name}.mid",
                data=p.read_bytes(),
                file_name=f"{gen_dir.name}_{name}.mid",
                mime="audio/midi",
                key=f"dl_gen_{gen_dir.name}_{name}",
            )

    st.download_button(
        label="Download generation_output.zip",
        data=_zip_dir(gen_dir),
        file_name=f"generation_{gen_dir.name}.zip",
        mime="application/zip",
        key=f"dl_gen_zip_{gen_dir.name}",
    )

    preview = gen_dir / "preview.wav"
    if preview.is_file():
        st.audio(str(preview))
    else:
        st.info("No preview.wav (preview_unavailable) — open song.mid in a DAW.")


def _render_generation_panel(analysis_output_dir: Path) -> None:
    st.divider()
    st.subheader("Generate new track (Block B)")
    st.caption(
        "Uses only analysis_output/ + your prompt. Never re-reads the reference audio."
    )

    api_ok = has_api_key()
    gen_root = analysis_output_dir.parent / "generation_output"

    prompt = st.text_input(
        "Generation prompt",
        value=st.session_state.get("gen_prompt", ""),
        placeholder="calm exploration theme, darker than the reference",
        disabled=not api_ok,
        key="gen_prompt_input",
    )
    st.session_state.gen_prompt = prompt

    col_a, col_b = st.columns(2)
    gen_clicked = col_a.button(
        "Generate",
        type="primary",
        disabled=not api_ok,
        key="btn_generate",
    )
    regen_clicked = col_b.button(
        "Regenerate",
        disabled=not api_ok,
        key="btn_regenerate",
        help="New variation folder; previous generations stay listed",
    )
    if not api_ok:
        st.info("no API key configured — generation disabled (existing results still listed)")

    if gen_clicked or regen_clicked:
        progress = st.container()
        try:
            with progress.status("11 · PLAN", expanded=True) as s:
                s.write("calling model…")
            # generate_track does plan+parts+render+preview internally;
            # show staged statuses around the call for UX.
            with progress.status("11–14 · PLAN → PARTS → RENDER → PREVIEW", expanded=True) as s:
                s.write(f"prompt={prompt or 'same vibe as the reference'!r}")
                result = generate_track(
                    analysis_output_dir,
                    user_prompt=prompt or "",
                    output_dir=gen_root,
                    skip_preview=False,
                )
                s.write(f"→ {result.get('output_dir')}")
                s.update(label="11–14 · Generation ✓", state="complete")
            st.session_state.last_generation = result.get("output_dir")
            st.session_state.gen_error = None
            st.success(f"Generated → {result.get('output_dir')}")
        except Exception as exc:  # noqa: BLE001
            # Keep the full traceback in the terminal; the UI needs an actionable error.
            traceback.print_exc()
            st.session_state.gen_error = str(exc)
            st.error("Generation failed (analysis + prior generations still OK)")
            st.code(st.session_state.gen_error)

    if st.session_state.get("gen_error") and not st.session_state.get("last_generation"):
        pass  # already shown on click

    completed = list_completed_generations(gen_root)
    if not completed:
        st.write("No completed generations yet.")
        return

    labels = [p.name for p in completed]
    default_ix = len(labels) - 1
    last = st.session_state.get("last_generation")
    if last:
        last_name = Path(last).name
        if last_name in labels:
            default_ix = labels.index(last_name)

    choice = st.selectbox(
        "Completed generations",
        options=labels,
        index=default_ix,
        key=f"gen_select_{analysis_output_dir}",
    )
    chosen = gen_root / choice
    _render_one_generation(chosen)


def main() -> None:
    st.set_page_config(page_title="flithack", layout="wide")
    st.title("flithack — reference → analysis → new MIDI")
    st.caption("Block A analysis + Block B generation (prompt-inspired, editable MIDI)")

    api_ok = has_api_key()
    if "run_llm" not in st.session_state:
        st.session_state.run_llm = api_ok

    run_llm = st.checkbox(
        "Run LLM interpretation",
        value=st.session_state.run_llm if api_ok else False,
        disabled=not api_ok,
        help="Requires OPENAI_API_KEY in environment / .env",
        key="run_llm_checkbox",
    )
    if not api_ok:
        st.info("no API key configured — LLM interpretation disabled")
    st.session_state.run_llm = run_llm if api_ok else False

    uploaded = st.file_uploader(
        "Drop reference audio",
        type=["mp3", "wav"],
        accept_multiple_files=False,
    )
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    current_upload_sha = (
        _sha256_bytes(uploaded_bytes) if uploaded_bytes is not None else None
    )
    force = st.checkbox("Force recompute (ignore stage caches)", value=False)

    analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

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
        st.session_state.pipeline_running = True

        progress = st.container()
        result = _run_pipeline(
            input_path,
            output_dir,
            work_dir,
            run_llm=bool(st.session_state.run_llm),
            force=force,
            status_container=progress,
        )
        st.session_state.pipeline_running = False
        st.session_state.result_ok = result.get("ok", False)
        st.session_state.result_error = result.get("error")
        st.session_state.midi_text = result.get("midi_text")
        st.session_state.interp = result.get("interp")

        if not result.get("ok"):
            st.error("Pipeline failed")
            st.code(result.get("error") or "unknown error")
        else:
            st.success(f"Done → {output_dir}")

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
            st.code(st.session_state.get("result_error"))
        _render_results(
            Path(out),
            st.session_state.get("midi_text"),
            st.session_state.get("interp"),
        )
    elif uploaded is None:
        st.write("Upload an mp3/wav and click **Analyze**.")
    else:
        st.info("New audio selected — click **Analyze** to see its results.")


if __name__ == "__main__":
    main()
