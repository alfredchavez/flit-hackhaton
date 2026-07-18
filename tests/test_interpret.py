import json
import tempfile
import unittest
from pathlib import Path

from flithack.interpret import (
    input_sha256,
    interpret_analysis,
    load_valid_cached_interpretation,
    prompt_sha256,
)


class InterpretationCacheTests(unittest.TestCase):
    def _write_cache(self, root: Path, midi_text: str) -> Path:
        path = root / "llm_interpretation.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "0.1",
                    "model": "gpt-5-mini",
                    "prompt_sha256": prompt_sha256(),
                    "input_sha256": input_sha256(midi_text),
                }
            )
        )
        return path

    def test_valid_cache_is_available_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cache(root, "current")

            result = interpret_analysis(
                root, skip_network=True, midi_text="current"
            )

            self.assertTrue(result["ok"])
            self.assertTrue(result["cached"])
            self.assertIsNotNone(
                load_valid_cached_interpretation(root, midi_text="current")
            )

    def test_invalid_cache_is_removed_when_network_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_cache(root, "old")

            result = interpret_analysis(root, skip_network=True, midi_text="new")

            self.assertTrue(result["skipped"])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
