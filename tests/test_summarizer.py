import os
import unittest
from unittest.mock import MagicMock, patch

from google.genai import errors
import httpx

import summarizer


class SummarizerTests(unittest.TestCase):
    def test_missing_api_key_is_reported_without_creating_client(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
            patch.object(summarizer.genai, "Client") as client,
        ):
            with self.assertRaisesRegex(
                summarizer.SummarizationError, "GEMINI_API_KEY"
            ):
                summarizer.summarize_letter("ಪತ್ರದ ಪಠ್ಯ")
        client.assert_not_called()

    def test_empty_ocr_text_does_not_create_client(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(summarizer.genai, "Client") as client,
        ):
            with self.assertRaisesRegex(
                summarizer.SummarizationError, "OCR"
            ):
                summarizer.summarize_letter(" \n\t")
        client.assert_not_called()

    def test_prompt_covers_required_fields_and_hallucination_safeguards(self):
        for required in (
            "ಗಡುವುಗಳು ಮತ್ತು ದಿನಾಂಕಗಳು",
            "ಮೇಲ್ಮನವಿ ಮತ್ತು ದೂರು ಸಲ್ಲಿಸುವ ವಿಧಾನ",
            "ಮಾಡಬೇಕಾದ ಮುಂದಿನ ಕ್ರಮಗಳು",
            "ಅಗತ್ಯ ದಾಖಲೆಗಳು ಮತ್ತು ಷರತ್ತುಗಳು",
            "ಉಲ್ಲೇಖ ಸಂಖ್ಯೆಗಳು ಮತ್ತು ಸಂಬಂಧಿತ ಅಧಿಕಾರಿಗಳು",
            "ಪತ್ರದಲ್ಲಿ ಉಲ್ಲೇಖಿಸಿಲ್ಲ",
            "Never invent or infer",
            "verbatim Kannada quotations",
            "contradictory",
        ):
            with self.subTest(required=required):
                self.assertIn(required, summarizer.SYSTEM_PROMPT)

    def test_api_error_is_reported_without_crashing(self):
        api_error = errors.APIError(429, {"message": "quota"})
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.side_effect = api_error
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(summarizer.genai, "Client", return_value=client),
        ):
            with self.assertRaisesRegex(summarizer.SummarizationError, "ಮಿತಿ"):
                summarizer.summarize_letter("ಪತ್ರದ ಪಠ್ಯ")

    def test_connection_error_is_reported_without_crashing(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.side_effect = httpx.ConnectError(
            "connection failed"
        )
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(summarizer.genai, "Client", return_value=client),
        ):
            with self.assertRaisesRegex(
                summarizer.SummarizationError, "ಸಂಪರ್ಕ"
            ):
                summarizer.summarize_letter("ಪತ್ರದ ಪಠ್ಯ")

    def test_generated_summary_is_returned(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value.text = "ಸಾರಾಂಶದ ಪಠ್ಯ"
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(summarizer.genai, "Client", return_value=client),
        ):
            result = summarizer.summarize_letter("ಪತ್ರದ ಪಠ್ಯ")

        self.assertEqual(result, "ಸಾರಾಂಶದ ಪಠ್ಯ")
        client.models.generate_content.assert_called_once()

    def test_empty_api_response_is_reported(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value.text = None
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(summarizer.genai, "Client", return_value=client),
        ):
            with self.assertRaisesRegex(
                summarizer.SummarizationError, "ಓದಲು ಸಾಧ್ಯವಾದ ಸಾರಾಂಶ"
            ):
                summarizer.summarize_letter("ಪತ್ರದ ಪಠ್ಯ")

    def test_invalid_api_response_is_reported(self):
        class InvalidResponse:
            @property
            def text(self):
                raise ValueError("invalid response")

        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = InvalidResponse()
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(summarizer.genai, "Client", return_value=client),
        ):
            with self.assertRaisesRegex(
                summarizer.SummarizationError, "ಪ್ರತಿಕ್ರಿಯೆಯನ್ನು ಓದಲು"
            ):
                summarizer.summarize_letter("ಪತ್ರದ ಪಠ್ಯ")

    def test_question_answer_uses_letter_and_question_in_gemini_request(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value.text = "ಪತ್ರದಲ್ಲಿ ನೀಡಿದ ಉತ್ತರ"
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(summarizer.genai, "Client", return_value=client),
        ):
            result = summarizer.answer_letter_question(
                "ಪತ್ರದ ಪಠ್ಯ: ದಿನಾಂಕ 12 ನೇ ನವೆಂಬರ್",
                "ಕಡೆಯ ದಿನಾಂಕ ಯಾವುದು?",
            )

        self.assertEqual(result, "ಪತ್ರದಲ್ಲಿ ನೀಡಿದ ಉತ್ತರ")
        request = client.models.generate_content.call_args.kwargs
        self.assertIn("ಪತ್ರದ ಪಠ್ಯ: ದಿನಾಂಕ 12 ನೇ ನವೆಂಬರ್", request["contents"])
        self.assertIn("ಕಡೆಯ ದಿನಾಂಕ ಯಾವುದು?", request["contents"])
        self.assertIn("using only", summarizer.QUESTION_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
