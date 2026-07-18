import unittest

import pretty_midi

from flithack.midi_repr import _serialize_drums_grid, _serialize_other


class MidiRepresentationTests(unittest.TestCase):
    def test_drum_grid_uses_note_onset(self) -> None:
        note = pretty_midi.Note(velocity=100, pitch=36, start=0.125, end=0.25)

        text = _serialize_drums_grid([note], [(0.0, 2.0)], [0])

        self.assertIn("KICK   .x..............", text)

    def test_other_includes_notes_sustained_into_bar(self) -> None:
        note = pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=4.0)

        text = _serialize_other([note], [(0.0, 2.0), (2.0, 4.0)], [0, 1])

        self.assertIn("other bar 2: {C} span C4-C4, 1 notes", text)


if __name__ == "__main__":
    unittest.main()
