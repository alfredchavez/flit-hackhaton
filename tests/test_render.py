"""Renderer gates: fixtures and edge cases produce valid, aligned MIDI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mido
import pretty_midi

from flithack.generate import normalize_parts, normalize_plan
from flithack.render import parse_chord_symbol, render

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> tuple[dict, dict]:
    data = json.loads((FIXTURES / name).read_text())
    return data["plan"], data["parts"]


class RendererTests(unittest.TestCase):
    def test_clean_fixture_renders_aligned_parts(self) -> None:
        plan_raw, parts_raw = _load("generation_clean.json")
        plan = normalize_plan(plan_raw)
        parts = normalize_parts(parts_raw, plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render(plan, parts, root)
            expected_seconds = sum(s["bars"] for s in plan["sections"]) * 4 * 60 / plan["bpm"]

            for name in ("drums", "bass", "harmony", "melody"):
                path = root / "midi" / f"{name}.mid"
                self.assertTrue(path.is_file() and path.stat().st_size > 0)
                pm = pretty_midi.PrettyMIDI(str(path))
                self.assertAlmostEqual(pm.get_end_time(), expected_seconds, places=5)

            song = pretty_midi.PrettyMIDI(str(root / "song.mid"))
            self.assertEqual(len(song.instruments), 4)
            self.assertGreater(sum(len(i.notes) for i in song.instruments), 20)
            self.assertAlmostEqual(song.get_end_time(), expected_seconds, places=5)

    def test_messy_fixture_never_raises(self) -> None:
        plan_raw, parts_raw = _load("generation_messy.json")
        plan = normalize_plan(plan_raw, reference_key="A minor")
        parts = normalize_parts(parts_raw, plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warnings = render(plan, parts, root)
            self.assertEqual(plan["meter"], "4/4")
            self.assertTrue(40 <= plan["bpm"] <= 240)
            self.assertTrue(warnings)
            for name in ("drums", "bass", "harmony", "melody"):
                pretty_midi.PrettyMIDI(str(root / "midi" / f"{name}.mid"))
            pretty_midi.PrettyMIDI(str(root / "song.mid"))

    def test_pitched_note_can_sustain_across_barline(self) -> None:
        plan = {
            "bpm": 120,
            "key": "C major",
            "sections": [
                {
                    "id": "a",
                    "bars": 16,
                    "active_parts": ["bass"],
                    "chords": ["C"] * 16,
                }
            ],
        }
        parts = {
            "drums": [],
            "bass": [
                {
                    "section_id": "a",
                    "events_by_bar": ["C2@4.0 len2.0 accent"],
                }
            ],
            "melody": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render(plan, parts, root)
            bass = pretty_midi.PrettyMIDI(str(root / "midi" / "bass.mid"))
            self.assertAlmostEqual(bass.instruments[0].notes[0].end - bass.instruments[0].notes[0].start, 1.0)

    def test_uppercase_major_quality_is_not_minor(self) -> None:
        self.assertEqual(parse_chord_symbol("CM")[1], [0, 4, 7])
        self.assertEqual(parse_chord_symbol("CM7")[1], [0, 4, 7, 11])
        self.assertEqual(parse_chord_symbol("Cm7")[1], [0, 3, 7, 10])

    def test_empty_active_parts_are_aligned_and_warned(self) -> None:
        plan = {
            "bpm": 120,
            "key": "C major",
            "sections": [
                {
                    "id": "a",
                    "bars": 16,
                    "active_parts": ["drums", "bass", "melody"],
                    "chords": ["C"] * 16,
                }
            ],
        }
        parts = {
            "drums": [],
            "bass": [{"section_id": "a", "events_by_bar": [""]}],
            "melody": [{"section_id": "a", "events_by_bar": ["bad token"]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warnings = render(plan, parts, root)
            self.assertIn("part_dropped:drums:a", warnings)
            self.assertIn("part_dropped:bass:a", warnings)
            self.assertIn("part_dropped:melody:a", warnings)

            for name in ("drums", "bass", "harmony", "melody"):
                path = root / "midi" / f"{name}.mid"
                pm = pretty_midi.PrettyMIDI(str(path))
                self.assertAlmostEqual(pm.get_end_time(), 32.0, places=5)
                raw = mido.MidiFile(path)
                self.assertGreaterEqual(len(raw.tracks), 2)


if __name__ == "__main__":
    unittest.main()
