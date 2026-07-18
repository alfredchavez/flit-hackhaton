"""LLM interpretation of MIDI text → analysis_output/llm_interpretation.json."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROMPT_VERSION = "0.1"

SYSTEM_PROMPT = """You are a music producer reading a compact transcription of a reference track.

The text was produced by automatic analysis — not a finished score. Drums are shown as instrument step grids (KICK/SNARE/HAT/TOM/CYMBAL), not MIDI pitch numbers. Pitched parts use note names with beat positions inside each bar. "other" is a messy harmonic sketch (pitch-class sets), not a full arrangement.

Your job:
- Describe the musical character so a downstream composer/generator can build something *inspired by* this reference.
- Give concrete, reusable generation_hints (e.g. "syncopated 16th-note hats with accents on the off-beat of 4", "bass roots on downbeats with approach notes on the 'and' of 2").
- Do NOT reproduce the reference note-for-note or invent a full new arrangement.
- Be concrete, not generic. Avoid empty phrases like "energetic drums" without rhythm detail.
- The MIDI text has rhythm, pitch, harmony density, and velocity — NOT production timbre, mix, or real instruments. Do not invent sound design, synth brands, or confident genre labels that are not supported by the supplied representation.
- If data is sparse or unreliable (see warnings), say so briefly and still give best-effort structural traits.
"""


class DrumInterpretation(BaseModel):
    groove_description: str = Field(
        description="Concrete groove description, e.g. four-on-the-floor kick, backbeat snare"
    )
    feel: str = Field(description="straight / swung / shuffled (or closest fit)")
    density: str = Field(description="sparse / medium / dense")
    signature_elements: list[str] = Field(
        description="Distinctive drum features worth reusing"
    )


class Interpretation(BaseModel):
    structural_traits: list[str] = Field(
        description="Short tags e.g. minor-key, syncopated, four-on-the-floor"
    )
    overall_character: str
    drums: DrumInterpretation
    bass_behavior: str = Field(
        description="Rhythm, contour, relation to kick/roots"
    )
    melody_behavior: str = Field(
        description="Range, phrasing, contour, repetition"
    )
    harmony_color: str = Field(
        description="Chord qualities, movement, mood"
    )
    energy_arc: str = Field(
        description="How intensity evolves across the song"
    )
    generation_hints: list[str] = Field(
        description="Concrete advice for composing something inspired by this"
    )


def prompt_sha256() -> str:
    """Hash of system prompt + schema field names (prompt iteration invalidates cache)."""
    schema_blob = json.dumps(Interpretation.model_json_schema(), sort_keys=True)
    payload = SYSTEM_PROMPT + "\n---\n" + schema_blob + "\n---\n" + PROMPT_VERSION
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_sha256(midi_text: str) -> str:
    return hashlib.sha256(midi_text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".llm_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _update_profile_warning(
    profile_path: Path,
    *,
    add: str | None = None,
    remove: str | None = None,
) -> None:
    """Atomically add/remove a warning on reference_profile.json."""
    if not profile_path.is_file():
        return
    data = json.loads(profile_path.read_text())
    warnings = list(data.get("warnings") or [])
    if remove and remove in warnings:
        warnings = [w for w in warnings if w != remove]
    if add and add not in warnings:
        warnings.append(add)
    data["warnings"] = warnings
    _atomic_write_json(profile_path, data)


def _cache_valid(
    path: Path,
    *,
    model: str,
    p_sha: str,
    i_sha: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("model") != model:
        return None
    if data.get("prompt_sha256") != p_sha:
        return None
    if data.get("input_sha256") != i_sha:
        return None
    if data.get("schema_version") != "0.1":
        return None
    return data


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def load_valid_cached_interpretation(
    analysis_output_dir: str | Path,
    *,
    midi_text: str | None = None,
) -> dict[str, Any] | None:
    """Return the on-disk interpretation only when its provenance is current."""
    root = Path(analysis_output_dir)
    if midi_text is None:
        from flithack.midi_repr import midi_repr

        try:
            midi_text = midi_repr(root)
        except Exception:  # noqa: BLE001 — an unreadable cache is simply unusable
            return None
    return _cache_valid(
        root / "llm_interpretation.json",
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        p_sha=prompt_sha256(),
        i_sha=input_sha256(midi_text),
    )


def interpret_analysis(
    analysis_output_dir: str | Path,
    *,
    force: bool = False,
    skip_network: bool = False,
    midi_text: str | None = None,
) -> dict[str, Any]:
    """
    Build midi text, call OpenAI (unless cached/skipped), write llm_interpretation.json.

    Never raises for API failures — marks llm_interpretation_failed and continues.
    Public stage entrypoint.
    """
    root = Path(analysis_output_dir)
    profile_path = root / "reference_profile.json"
    out_path = root / "llm_interpretation.json"
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    p_sha = prompt_sha256()

    result: dict[str, Any] = {
        "ok": False,
        "skipped": False,
        "cached": False,
        "path": str(out_path),
        "error": None,
        "interpretation": None,
    }

    if midi_text is None:
        from flithack.midi_repr import midi_repr

        try:
            midi_text = midi_repr(root)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"midi_repr failed: {exc}"
            out_path.unlink(missing_ok=True)
            if not skip_network:
                _update_profile_warning(profile_path, add="llm_interpretation_failed")
            print(f"[interpret] midi_repr failed: {exc}")
            return result

    i_sha = input_sha256(midi_text)
    cached = _cache_valid(out_path, model=model, p_sha=p_sha, i_sha=i_sha)

    if not force and cached is not None:
        _update_profile_warning(profile_path, remove="llm_interpretation_failed")
        result["ok"] = True
        result["cached"] = True
        result["interpretation"] = cached
        print(f"[interpret] cache hit → {out_path}")
        return result

    # Never leave a provenance-invalid artifact available for the UI or ZIP.
    if cached is None:
        out_path.unlink(missing_ok=True)

    if skip_network:
        result["skipped"] = True
        result["error"] = "skipped (no network / user disabled)"
        print("[interpret] skipped (network disabled)")
        return result

    if not has_api_key():
        result["skipped"] = True
        result["error"] = "no API key configured"
        print("[interpret] no API key configured — skipping")
        return result

    # A forced refresh must not silently fall back to the superseded result.
    if force:
        out_path.unlink(missing_ok=True)

    try:
        from openai import OpenAI

        client = OpenAI(timeout=30.0, max_retries=1)
        rsp = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": midi_text},
            ],
            text_format=Interpretation,
        )
        interp = rsp.output_parsed
        if interp is None:
            raise RuntimeError("OpenAI returned no parsed interpretation")

        payload = interp.model_dump()
        payload.update(
            {
                "schema_version": "0.1",
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": p_sha,
                "input_sha256": i_sha,
            }
        )
        _atomic_write_json(out_path, payload)
        _update_profile_warning(profile_path, remove="llm_interpretation_failed")
        result["ok"] = True
        result["interpretation"] = payload
        print(f"[interpret] wrote {out_path}")
        return result
    except Exception as exc:  # noqa: BLE001 — never fail pipeline
        result["error"] = str(exc)
        _update_profile_warning(profile_path, add="llm_interpretation_failed")
        print(f"[interpret] failed (continuing): {exc}")
        return result
