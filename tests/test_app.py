import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from summarizer import SummarizationError
import tts


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def test_microphone_is_hidden_before_summary_exists(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)

        self.assertEqual(len(app.audio_input), 0)

    def test_existing_ocr_runs_on_request_and_is_not_repeated_for_summary(self):
        image_buffer = BytesIO()
        Image.new("RGB", (1, 1)).save(image_buffer, format="PNG")
        ocr_result = {
            "text": "ಪತ್ರದ ಪಠ್ಯ",
            "words": [],
            "detected_languages": {},
        }

        with (
            patch("ocr.prepare_image", return_value=b"prepared"),
            patch("ocr.run_ocr", return_value=ocr_result) as run_ocr,
            patch("summarizer.summarize_letter", return_value="ಕನ್ನಡ ಸಾರಾಂಶ"),
            patch("tts.synthesize_speech", return_value=b"audio"),
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
            app.file_uploader[0].upload(
                "letter.png", image_buffer.getvalue(), "image/png"
            )
            app.run(timeout=15)
            self.assertEqual(
                [button.label for button in app.button], ["ಪಠ್ಯ ಓದಿ"]
            )
            run_ocr.assert_not_called()

            app.button[0].click().run(timeout=15)
            run_ocr.assert_called_once_with(b"prepared")

            app.button[1].click().run(timeout=15)
            run_ocr.assert_called_once()

    def test_summary_is_displayed_and_downloadable(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.session_state["ocr_result"] = {
            "text": "ಪತ್ರದ ಪಠ್ಯ",
            "words": [],
            "detected_languages": {},
        }
        app.run(timeout=15)

        with (
            patch("summarizer.summarize_letter", return_value="ಕನ್ನಡ ಸಾರಾಂಶ"),
            patch("tts.synthesize_speech", return_value=b"audio"),
        ):
            app.button[0].click().run(timeout=15)

        self.assertIn("ಕನ್ನಡ ಸಾರಾಂಶ", [item.value for item in app.markdown])
        download_labels = [button.label for button in app.get("download_button")]
        self.assertIn("ಸಾರಾಂಶ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ", download_labels)

    def test_microphone_is_shown_after_summary_exists(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.session_state["ocr_result"] = {
            "text": "ಪತ್ರದ ಪಠ್ಯ",
            "words": [],
            "detected_languages": {},
        }
        app.session_state["summary_result"] = "ಕನ್ನಡ ಸಾರಾಂಶ"
        with patch("tts.synthesize_speech", return_value=b"audio"):
            app.run(timeout=15)

        self.assertEqual(len(app.audio_input), 1)

    def test_summary_error_is_shown_in_streamlit(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.session_state["ocr_result"] = {
            "text": "ಪತ್ರದ ದೋಷ ಪರೀಕ್ಷೆ",
            "words": [],
            "detected_languages": {},
        }
        app.run(timeout=15)

        with patch(
            "summarizer.summarize_letter",
            side_effect=SummarizationError("GEMINI_API_KEY ಹೊಂದಿಸಿ"),
        ):
            app.button[0].click().run(timeout=15)

        self.assertIn("GEMINI_API_KEY ಹೊಂದಿಸಿ", [message.value for message in app.error])

    def test_summary_tts_uses_normalized_text(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.session_state["ocr_result"] = {
            "text": "ಪತ್ರದ ಪಠ್ಯ",
            "words": [],
            "detected_languages": {},
        }
        app.session_state["summary_result"] = "ವಿಶಿಷ್ಟ ಮೂಲ ಸಾರಾಂಶ 7 ದಿನಗಳು"
        with (
            patch(
                "tts_normalizer.normalize_for_tts",
                return_value="ವಿಶಿಷ್ಟ ಮಾತಿನ ಸಾರಾಂಶ",
            ),
            patch("tts.synthesize_speech", return_value=b"audio") as synthesize,
        ):
            app.run(timeout=15)
            next(button for button in app.button if button.label == "ಕೇಳಿ").click()
            app.run(timeout=15)

        synthesize.assert_called_once_with(
            "ವಿಶಿಷ್ಟ ಮಾತಿನ ಸಾರಾಂಶ", voice="Shivani", speed=1.0
        )

    def test_answer_tts_uses_normalized_text(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.session_state["ocr_result"] = {
            "text": "ಪತ್ರದ ಪಠ್ಯ",
            "words": [],
            "detected_languages": {},
        }
        app.session_state["summary_result"] = "ಸಾರಾಂಶ"
        app.session_state["qa_ocr_text"] = "ಪತ್ರದ ಪಠ್ಯ"
        app.session_state["conversation"] = [
            {"question": "ಪ್ರಶ್ನೆ", "answer": "ಮೂಲ ಉತ್ತರ 10:00"}
        ]
        with (
            patch(
                "tts_normalizer.normalize_for_tts",
                side_effect=lambda text: (
                    "ಮಾತಿನ ಸಾರಾಂಶ" if text == "ಸಾರಾಂಶ" else "ಮಾತಿನ ಪಠ್ಯ"
                ),
            ),
            patch("tts.synthesize_speech", return_value=b"audio") as synthesize,
        ):
            app.run(timeout=15)
            next(button for button in app.button if button.label == "ಉತ್ತರ ಕೇಳಿ").click()
            app.run(timeout=15)

        self.assertEqual(synthesize.call_count, 2)
        synthesize.assert_any_call("ಮಾತಿನ ಪಠ್ಯ", voice="Shivani", speed=1.0)


if __name__ == "__main__":
    unittest.main()
