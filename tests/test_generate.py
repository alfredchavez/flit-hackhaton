import os
import unittest
from unittest.mock import patch

import httpx
from openai import APITimeoutError

from flithack.generate import (
    GenerationPlan,
    GenerationTimeoutError,
    _openai_parse,
    generation_read_timeout_seconds,
)


class GenerationRequestTests(unittest.TestCase):
    def test_generation_timeout_is_configurable_and_bounded(self) -> None:
        with patch.dict(os.environ, {"GENERATION_TIMEOUT_SECONDS": "180"}):
            self.assertEqual(generation_read_timeout_seconds(), 180.0)
        with patch.dict(os.environ, {"GENERATION_TIMEOUT_SECONDS": "invalid"}):
            self.assertEqual(generation_read_timeout_seconds(), 300.0)
        with patch.dict(os.environ, {"GENERATION_TIMEOUT_SECONDS": "9999"}):
            self.assertEqual(generation_read_timeout_seconds(), 600.0)

    def test_timeout_error_names_stage_and_configuration(self) -> None:
        timeout = APITimeoutError(
            request=httpx.Request("POST", "https://api.openai.com/v1/responses")
        )
        with patch.dict(os.environ, {"GENERATION_TIMEOUT_SECONDS": "75"}):
            with patch("openai.OpenAI") as client_class:
                client_class.return_value.responses.parse.side_effect = timeout

                with self.assertRaisesRegex(
                    GenerationTimeoutError,
                    "PLAN timed out.*GENERATION_TIMEOUT_SECONDS",
                ):
                    _openai_parse(
                        "test-model",
                        "system",
                        "user",
                        GenerationPlan,
                        stage="PLAN",
                    )

                configured = client_class.call_args.kwargs["timeout"]
                self.assertEqual(configured.read, 75.0)
                self.assertEqual(configured.connect, 10.0)
                self.assertEqual(client_class.call_args.kwargs["max_retries"], 1)


if __name__ == "__main__":
    unittest.main()
