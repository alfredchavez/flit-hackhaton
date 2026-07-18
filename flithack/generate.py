"""Block B: PLAN + PARTS LLM calls → render → generation_output/."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from flithack.interpret import load_valid_cached_interpretation
from flithack.preview import render_preview
from flithack.render import key_tonic_triad, render

PLAN_PROMPT_VERSION = "0.1"
PARTS_PROMPT_VERSION = "0.1"

ALLOWED_PARTS = ("drums", "bass", "harmony", "melody")
DEFAULT_GENERATION_READ_TIMEOUT_SECONDS = 300.0


class GenerationTimeoutError(RuntimeError):
    """A PLAN or PARTS request exceeded the configured read deadline."""


# ── Pydantic schemas ─────────────────────────────────────────────────────────


class Section(BaseModel):
    id: str = Field(description="unique stable ID: intro / groove_a / groove_b / outro")
    name: str = Field(description="human label shown in the UI")
    bars: int = Field(description="2–16")
    energy: float = Field(description="0–1")
    active_parts: list[str] = Field(
        description="subset of: drums, bass, harmony, melody"
    )
    chords: list[str] = Field(
        description='one symbol per bar: "Fm", "Ab", "Cm7", "Bb"'
    )


class GenerationPlan(BaseModel):
    title: str
    bpm: float = Field(description="near reference unless the prompt says otherwise")
    key: str = Field(description='e.g. "C minor" — may differ from reference')
    meter: str = Field(description='requested value; normalizer always forces "4/4"')
    sections: list[Section] = Field(description="total 16–32 bars")
    style_notes: str = Field(description="carried verbatim into the PARTS call")


class DrumPattern(BaseModel):
    section_id: str
    kick: str = Field(description="exactly 16 chars of x / o / .")
    snare: str
    closed_hat: str
    tom: str
    cymbal: str
    fill_last_bar: bool = False


class Phrase(BaseModel):
    section_id: str
    events_by_bar: list[str] = Field(
        description='1–4 bar strings: "F2@1.0 len1.0 accent | Ab2@3.5 len0.5 soft"'
    )


class Parts(BaseModel):
    drums: list[DrumPattern] = Field(
        description="one per section where drums are active"
    )
    bass: list[Phrase] = Field(description="one phrase per active section")
    melody: list[Phrase] = Field(description="one or two per active section")


PLAN_SYSTEM = """You are a game-audio composer planning a NEW track inspired by a reference analysis.

The analysis describes a *reference* only. Borrow its musical language (tempo feel, chord colors, groove density, energy arc) — do NOT reproduce exact melodies, basslines, or drum hooks from the reference. The reference provides vocabulary, not content.

Output a compact loop-friendly plan: 16–32 bars total, 4/4, clear sections with chord symbols (one per bar), energy 0–1, and which parts are active (drums, bass, harmony, melody).

Stay near the reference BPM/key unless the user prompt explicitly asks otherwise (e.g. "faster", "darker", "in D minor").
Section ids must be unique and stable (intro, groove_a, groove_b, bridge, outro, …).
"""

PARTS_SYSTEM = """You are a game-audio composer writing NEW drum grids and bass/melody phrases for a planned track.

You receive:
1) A normalized generation plan (authoritative structure).
2) Compact style/interpretation notes from a reference analysis.
3) Representative bars of the reference transcription — these are FORMAT and STYLE evidence only. Do NOT copy those notes, riffs, or drum hooks. Invent new material in a similar language.

Drums: each pattern is 16-step strings using only x (accent), o (soft), . (rest) for kick/snare/closed_hat/tom/cymbal.
Pitched events grammar (exactly):
  NOTE@BEAT lenDURATION VELOCITY
where BEAT is 1.0 <= beat < 5.0 (local to the bar), VELOCITY is accent|med|soft.
Example bar: F2@1.0 len1.0 accent | F2@2.5 len0.5 soft | Ab2@3.0 len1.0 med
events_by_bar is 1–4 bar strings; the renderer loops them across the section. Empty string = rest bar.

Provide drum patterns for every section where drums are active, and bass/melody phrases for every section where those parts are active. Harmony is built from the plan chords — do not output harmony.

Originality: new composition inspired by the reference vocabulary. Never paste the reference material.
"""


def generation_model() -> str:
    gen = os.getenv("GENERATION_MODEL")
    if gen is not None and gen.strip() == "":
        # explicitly empty → fall through to OPENAI_MODEL
        gen = None
    elif gen is not None:
        gen = gen.strip() or None
    return gen or os.getenv("OPENAI_MODEL", "gpt-5-mini")


def generation_read_timeout_seconds() -> float:
    """Configurable read timeout, bounded so a typo cannot hang forever."""
    raw = os.getenv(
        "GENERATION_TIMEOUT_SECONDS",
        str(DEFAULT_GENERATION_READ_TIMEOUT_SECONDS),
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_GENERATION_READ_TIMEOUT_SECONDS
    return max(30.0, min(600.0, value))


def generation_prompt_sha256() -> str:
    blob = (
        PLAN_SYSTEM
        + "\n---\n"
        + PARTS_SYSTEM
        + "\n---\n"
        + json.dumps(GenerationPlan.model_json_schema(), sort_keys=True)
        + "\n---\n"
        + json.dumps(Parts.model_json_schema(), sort_keys=True)
        + "\n---\n"
        + PLAN_PROMPT_VERSION
        + "\n"
        + PARTS_PROMPT_VERSION
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def analysis_sha256(
    profile: dict[str, Any],
    interpretation: dict[str, Any] | None,
    midi_text: str,
) -> str:
    payload = {
        "profile": profile,
        "interpretation": interpretation,  # null when absent
        "midi_repr": midi_text,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Compact reference profile for PLAN input."""
    chords = profile.get("chords") or []
    chord_seq: list[str] = []
    for c in chords:
        name = str(c.get("chord", "N"))
        if not chord_seq or chord_seq[-1] != name:
            chord_seq.append(name)
    if len(chord_seq) > 24:
        chord_seq = chord_seq[:12] + ["…"] + chord_seq[-8:]

    energy = profile.get("energy_curve") or []
    vals = [float(e.get("value", 0)) for e in energy]
    energy_summary = {
        "bars": len(vals),
        "avg": round(sum(vals) / len(vals), 3) if vals else 0,
        "peak": round(max(vals), 3) if vals else 0,
        "low": round(min(vals), 3) if vals else 0,
    }
    sections = profile.get("sections") or []
    return {
        "bpm": profile.get("bpm"),
        "key": profile.get("key"),
        "meter": profile.get("meter"),
        "duration_seconds": profile.get("duration_seconds"),
        "chord_summary": chord_seq,
        "energy": energy_summary,
        "sections": sections,
        "per_stem": profile.get("per_stem") or {},
        "warnings": profile.get("warnings") or [],
    }


def _parse_key(key: str) -> str | None:
    if not key or not str(key).strip():
        return None
    m = re.match(
        r"^\s*([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m)?\s*$",
        str(key),
        re.I,
    )
    if not m:
        return None
    root = m.group(1).upper() + (m.group(2) or "")
    mode = (m.group(3) or "major").lower()
    if mode in ("m", "min", "minor"):
        return f"{root} minor"
    return f"{root} major"


def normalize_plan(
    plan: dict[str, Any] | GenerationPlan,
    *,
    reference_key: str | None = None,
) -> dict[str, Any]:
    """Deterministic repairs. Renderer receives only the normalized plan."""
    if isinstance(plan, GenerationPlan):
        data = plan.model_dump()
    else:
        data = dict(plan)

    warnings: list[str] = []

    # BPM
    try:
        bpm = float(data.get("bpm") or 120.0)
    except (TypeError, ValueError):
        bpm = 120.0
        warnings.append("bpm_defaulted")
    if bpm < 40 or bpm > 240:
        warnings.append(f"bpm_clamped:{bpm}")
        bpm = max(40.0, min(240.0, bpm))
    data["bpm"] = bpm

    # Key
    key = _parse_key(str(data.get("key") or ""))
    if key is None:
        ref = _parse_key(str(reference_key or "")) if reference_key else None
        key = ref or "C major"
        warnings.append(f"key_fallback:{key}")
    data["key"] = key

    # Meter
    meter = str(data.get("meter") or "4/4").replace(" ", "")
    if meter != "4/4":
        warnings.append("meter_forced_4_4")
    data["meter"] = "4/4"

    tonic = key_tonic_triad(key)
    sections_in = list(data.get("sections") or [])
    sections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for i, sec in enumerate(sections_in):
        if not isinstance(sec, dict):
            try:
                sec = dict(sec)
            except Exception:
                continue
        sid = str(sec.get("id") or f"section_{i}").strip() or f"section_{i}"
        base = sid
        n = 2
        while sid in seen_ids:
            sid = f"{base}_{n}"
            n += 1
        seen_ids.add(sid)

        try:
            bars = int(sec.get("bars") or 4)
        except (TypeError, ValueError):
            bars = 4
        if bars < 2 or bars > 16:
            warnings.append(f"bars_clamped:{sid}:{bars}")
            bars = max(2, min(16, bars))

        try:
            energy = float(sec.get("energy") if sec.get("energy") is not None else 0.5)
        except (TypeError, ValueError):
            energy = 0.5
        energy = max(0.0, min(1.0, energy))

        parts = []
        for p in sec.get("active_parts") or []:
            p = str(p).lower().strip()
            if p in ALLOWED_PARTS and p not in parts:
                parts.append(p)
        if not parts:
            parts = list(ALLOWED_PARTS)
            warnings.append(f"active_parts_defaulted:{sid}")

        chords = [str(c).strip() for c in (sec.get("chords") or []) if str(c).strip()]
        if not chords:
            chords = [tonic]
            warnings.append(f"chords_defaulted_tonic:{sid}")
        # pad/truncate to bars
        if len(chords) < bars:
            last = chords[-1]
            chords = chords + [last] * (bars - len(chords))
            warnings.append(f"chords_padded:{sid}")
        elif len(chords) > bars:
            chords = chords[:bars]
            warnings.append(f"chords_truncated:{sid}")

        sections.append(
            {
                "id": sid,
                "name": str(sec.get("name") or sid),
                "bars": bars,
                "energy": energy,
                "active_parts": parts,
                "chords": chords,
            }
        )

    if not sections:
        sections = [
            {
                "id": "groove_a",
                "name": "Groove A",
                "bars": 16,
                "energy": 0.6,
                "active_parts": list(ALLOWED_PARTS),
                "chords": [tonic] * 16,
            }
        ]
        warnings.append("default_section_created")

    total = sum(s["bars"] for s in sections)
    # pad/repeat or truncate to 16–32
    if total < 16:
        # repeat last section bars or add bars to last
        need = 16 - total
        last = sections[-1]
        while need > 0:
            add = min(16 - last["bars"], need) if last["bars"] < 16 else 0
            if add > 0:
                last["bars"] += add
                last["chords"] = last["chords"] + [last["chords"][-1]] * add
                need -= add
            else:
                # duplicate last section with new id
                clone = dict(last)
                clone["id"] = f"{last['id']}_rep{len(sections)}"
                clone["name"] = last["name"] + " (rep)"
                take = min(clone["bars"], need)
                clone["bars"] = take
                clone["chords"] = clone["chords"][:take]
                sections.append(clone)
                need -= take
        warnings.append("form_padded_to_16")
    elif total > 32:
        # truncate from end
        keep = []
        acc = 0
        for s in sections:
            if acc >= 32:
                break
            if acc + s["bars"] <= 32:
                keep.append(s)
                acc += s["bars"]
            else:
                rem = 32 - acc
                if rem >= 2:
                    s = dict(s)
                    s["bars"] = rem
                    s["chords"] = s["chords"][:rem]
                    keep.append(s)
                    acc += rem
                break
        sections = keep
        warnings.append("form_truncated_to_32")

    data["sections"] = sections
    data["title"] = str(data.get("title") or "Untitled")
    data["style_notes"] = str(data.get("style_notes") or "")
    data["warnings"] = warnings
    data["schema_version"] = "0.1"
    return data


def normalize_parts(
    parts: dict[str, Any] | Parts,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Drop unknown sections, repair grids/events, cap phrases."""
    if isinstance(parts, Parts):
        data = parts.model_dump()
    else:
        data = {
            "drums": list(parts.get("drums") or []),
            "bass": list(parts.get("bass") or []),
            "melody": list(parts.get("melody") or []),
        }

    warnings: list[str] = []
    valid_ids = {s["id"] for s in plan.get("sections") or []}
    active_map = {
        s["id"]: set(s.get("active_parts") or []) for s in plan.get("sections") or []
    }

    # drums: first per section
    drums_out = []
    seen_d: set[str] = set()
    for d in data.get("drums") or []:
        if not isinstance(d, dict):
            d = dict(d)
        sid = d.get("section_id")
        if sid not in valid_ids:
            warnings.append(f"drop_drum_unknown_section:{sid}")
            continue
        if "drums" not in active_map.get(sid, set()):
            warnings.append(f"drop_drum_inactive:{sid}")
            continue
        if sid in seen_d:
            warnings.append(f"drop_extra_drum:{sid}")
            continue
        seen_d.add(sid)
        row = {}
        for k in ("kick", "snare", "closed_hat", "tom", "cymbal"):
            chars = [c for c in str(d.get(k) or "") if c in "xo.XO*"]
            chars = [("x" if c in "xX*" else "o" if c in "oO" else ".") for c in chars]
            if len(chars) < 16:
                if chars:
                    warnings.append(f"drum_grid_padded:{sid}:{k}")
                chars = chars + ["."] * (16 - len(chars))
            elif len(chars) > 16:
                warnings.append(f"drum_grid_truncated:{sid}:{k}")
                chars = chars[:16]
            row[k] = "".join(chars)
        drums_out.append(
            {
                "section_id": sid,
                **row,
                "fill_last_bar": bool(d.get("fill_last_bar")),
            }
        )

    def _norm_phrases(raw: list, name: str, max_per: int) -> list[dict]:
        out = []
        counts: dict[str, int] = {}
        for ph in raw or []:
            if not isinstance(ph, dict):
                ph = dict(ph)
            sid = ph.get("section_id")
            if sid not in valid_ids:
                warnings.append(f"drop_{name}_unknown_section:{sid}")
                continue
            if name not in active_map.get(sid, set()):
                warnings.append(f"drop_{name}_inactive:{sid}")
                continue
            counts[sid] = counts.get(sid, 0) + 1
            if counts[sid] > max_per:
                warnings.append(f"drop_extra_{name}:{sid}")
                continue
            ebb = list(ph.get("events_by_bar") or [])
            ebb = [str(x) for x in ebb]
            if len(ebb) < 1:
                ebb = [""]
            if len(ebb) > 4:
                warnings.append(f"phrase_truncated:{name}:{sid}")
                ebb = ebb[:4]
            out.append({"section_id": sid, "events_by_bar": ebb})
        return out

    bass_out = _norm_phrases(data.get("bass") or [], "bass", 1)
    melody_out = _norm_phrases(data.get("melody") or [], "melody", 2)

    return {
        "drums": drums_out,
        "bass": bass_out,
        "melody": melody_out,
        "warnings": warnings,
        "schema_version": "0.1",
    }


def _openai_parse(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    *,
    stage: str,
) -> BaseModel:
    from openai import APITimeoutError, OpenAI

    read_timeout = generation_read_timeout_seconds()
    client = OpenAI(
        timeout=httpx.Timeout(
            connect=10.0,
            read=read_timeout,
            write=30.0,
            pool=10.0,
        ),
        max_retries=1,
    )
    try:
        rsp = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=schema,
        )
    except APITimeoutError as exc:
        raise GenerationTimeoutError(
            f"{stage} timed out waiting for the model after {read_timeout:.0f}s. "
            "Retry Generate, or increase GENERATION_TIMEOUT_SECONDS in .env."
        ) from exc
    if rsp.output_parsed is None:
        raise RuntimeError("OpenAI returned no parsed result")
    return rsp.output_parsed


def call_plan(
    *,
    user_prompt: str,
    profile: dict[str, Any],
    interpretation: dict[str, Any] | None,
    model: str | None = None,
) -> GenerationPlan:
    model = model or generation_model()
    compact = compact_profile(profile)
    prompt_text = user_prompt.strip() or "same vibe as the reference"
    user = (
        f"USER PROMPT:\n{prompt_text}\n\n"
        f"REFERENCE PROFILE (compact):\n{json.dumps(compact, indent=2)}\n\n"
    )
    if interpretation:
        # strip provenance hashes for prompt brevity
        interp_view = {
            k: interpretation[k]
            for k in (
                "structural_traits",
                "overall_character",
                "drums",
                "bass_behavior",
                "melody_behavior",
                "harmony_color",
                "energy_arc",
                "generation_hints",
            )
            if k in interpretation
        }
        user += f"REFERENCE INTERPRETATION:\n{json.dumps(interp_view, indent=2)}\n"
    else:
        user += "REFERENCE INTERPRETATION: null\n"
    user += "\nPlan a NEW 16–32 bar loop-friendly track. Do not copy the reference."
    return _openai_parse(
        model,
        PLAN_SYSTEM,
        user,
        GenerationPlan,
        stage="PLAN",
    )  # type: ignore[return-value]


def call_parts(
    *,
    plan: dict[str, Any],
    profile: dict[str, Any],
    interpretation: dict[str, Any] | None,
    midi_text: str,
    variation_nonce: int = 1,
    model: str | None = None,
) -> Parts:
    model = model or generation_model()
    prompt_bits = [
        f"VARIATION NONCE: {variation_nonce} "
        f"(produce a different valid variation if nonce > 1)",
        f"STYLE NOTES FROM PLAN:\n{plan.get('style_notes', '')}",
        f"NORMALIZED PLAN:\n{json.dumps({k: plan[k] for k in plan if k != 'warnings'}, indent=2)}",
        f"REFERENCE PROFILE (compact):\n{json.dumps(compact_profile(profile), indent=2)}",
    ]
    if interpretation:
        prompt_bits.append(
            "INTERPRETATION HINTS:\n"
            + json.dumps(
                {
                    "overall_character": interpretation.get("overall_character"),
                    "generation_hints": interpretation.get("generation_hints"),
                    "drums": interpretation.get("drums"),
                    "bass_behavior": interpretation.get("bass_behavior"),
                    "melody_behavior": interpretation.get("melody_behavior"),
                },
                indent=2,
            )
        )
    prompt_bits.append(
        "REFERENCE MIDI REPR (format/style evidence ONLY — do not copy):\n" + midi_text
    )
    user = "\n\n".join(prompt_bits)
    return _openai_parse(
        model,
        PARTS_SYSTEM,
        user,
        Parts,
        stage="PARTS",
    )  # type: ignore[return-value]


def _next_run_number(gen_root: Path) -> int:
    gen_root.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in gen_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            nums.append(int(p.name))
        m = re.match(r"^(\d+)\.partial$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def list_completed_generations(gen_root: Path) -> list[Path]:
    """Numbered folders that contain generation_complete.json."""
    if not gen_root.is_dir():
        return []
    out = []
    for p in sorted(gen_root.iterdir(), key=lambda x: x.name):
        if p.is_dir() and p.name.isdigit() and (p / "generation_complete.json").is_file():
            out.append(p)
    return out


def write_complete_marker(out_dir: Path, warnings: list[str]) -> None:
    required = [
        "generation_plan.json",
        "midi/drums.mid",
        "midi/bass.mid",
        "midi/harmony.mid",
        "midi/melody.mid",
        "song.mid",
    ]
    files = {}
    for rel in required:
        p = out_dir / rel
        if not p.is_file():
            raise RuntimeError(f"missing required artifact: {rel}")
        files[rel] = _file_sha256(p)
    marker = {
        "schema_version": "0.1",
        "files": files,
        "warnings": warnings,
    }
    (out_dir / "generation_complete.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )


def generate_track(
    analysis_output_dir: str | Path,
    *,
    user_prompt: str = "",
    output_dir: str | Path | None = None,
    variation_nonce: int | None = None,
    model: str | None = None,
    skip_preview: bool = False,
) -> dict[str, Any]:
    """
    Full block B: PLAN → PARTS → render → preview → atomic complete folder.

    If output_dir is a parent `generation_output/`, creates `<n>.partial` then renames to `<n>/`.
    If output_dir points at a concrete destination, writes there (CLI convenience).
    """
    analysis_output_dir = Path(analysis_output_dir)
    profile_path = analysis_output_dir / "reference_profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing {profile_path}")

    profile = json.loads(profile_path.read_text())
    from flithack.midi_repr import midi_repr

    midi_text = midi_repr(analysis_output_dir)
    interpretation = load_valid_cached_interpretation(
        analysis_output_dir, midi_text=midi_text
    )

    model = model or generation_model()
    a_sha = analysis_sha256(profile, interpretation, midi_text)
    p_sha = generation_prompt_sha256()

    # destination layout
    if output_dir is None:
        gen_root = analysis_output_dir.parent / "generation_output"
        n = variation_nonce or _next_run_number(gen_root)
        partial = gen_root / f"{n}.partial"
        final = gen_root / str(n)
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        dest = partial
        atomic = True
    else:
        output_dir = Path(output_dir)
        # if ends with generation_output or is that folder name, use numbered
        if output_dir.name == "generation_output" or (
            not (output_dir / "generation_plan.json").exists()
            and output_dir.suffix != ".partial"
            and not any(output_dir.glob("midi/*.mid"))
        ):
            gen_root = output_dir
            n = variation_nonce or _next_run_number(gen_root)
            partial = gen_root / f"{n}.partial"
            final = gen_root / str(n)
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(parents=True)
            dest = partial
            atomic = True
        else:
            dest = output_dir
            dest.mkdir(parents=True, exist_ok=True)
            final = dest
            n = variation_nonce or 1
            atomic = False

    result: dict[str, Any] = {
        "ok": False,
        "output_dir": str(final),
        "run_number": n,
        "warnings": [],
        "error": None,
    }

    try:
        print(f"[generate] PLAN (model={model})…")
        raw_plan = call_plan(
            user_prompt=user_prompt,
            profile=profile,
            interpretation=interpretation,
            model=model,
        )
        plan = normalize_plan(raw_plan, reference_key=str(profile.get("key") or ""))
        print(f"[generate] plan: {plan.get('title')} bpm={plan['bpm']} key={plan['key']} "
              f"bars={sum(s['bars'] for s in plan['sections'])}")

        print(f"[generate] PARTS (nonce={n})…")
        raw_parts = call_parts(
            plan=plan,
            profile=profile,
            interpretation=interpretation,
            midi_text=midi_text,
            variation_nonce=n,
            model=model,
        )
        parts = normalize_parts(raw_parts, plan)

        print("[generate] RENDER…")
        render_warnings = render(plan, parts, dest)
        all_warnings = list(plan.get("warnings") or []) + list(parts.get("warnings") or []) + render_warnings

        preview_warning = None
        if not skip_preview:
            print("[generate] PREVIEW…")
            ok_prev, preview_warning = render_preview(dest / "song.mid", dest / "preview.wav")
            if not ok_prev and preview_warning:
                all_warnings.append(preview_warning)
        else:
            all_warnings.append("preview_unavailable:skipped")

        # de-dupe
        seen = set()
        uniq = []
        for w in all_warnings:
            if w not in seen:
                seen.add(w)
                uniq.append(w)

        plan_doc = {
            "schema_version": "0.1",
            "user_prompt": user_prompt.strip() or "same vibe as the reference",
            "variation_nonce": n,
            "model": model,
            "prompt_sha256": p_sha,
            "analysis_sha256": a_sha,
            "plan": plan,
            "parts": parts,
            "warnings": uniq,
            "plan_prompt_version": PLAN_PROMPT_VERSION,
            "parts_prompt_version": PARTS_PROMPT_VERSION,
        }
        (dest / "generation_plan.json").write_text(json.dumps(plan_doc, indent=2) + "\n")
        write_complete_marker(dest, uniq)

        if atomic:
            if final.exists():
                shutil.rmtree(final)
            os.replace(dest, final)

        result["ok"] = True
        result["output_dir"] = str(final)
        result["warnings"] = uniq
        result["plan"] = plan
        print(f"[generate] done → {final}")
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        print(f"[generate] failed: {exc}")
        # leave partial for debug; do not rename
        if atomic and dest.exists():
            # keep .partial
            pass
        raise
