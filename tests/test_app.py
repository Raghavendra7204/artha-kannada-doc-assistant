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
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
            app.file_uploader[0].upload(
                "letter.png", image_buffer.getvalue(), "image/png"
            )
            app.run(timeout=15)
            self.assertEqual(
                [button.label for button in app.button], ["Read text"]
            )
            run_ocr.assert_not_called()

            app.button[0].click().run(timeout=15)
            run_ocr.assert_called_once_with(b"prepared")

            with patch(
                "summarizer.summarize_letter", return_value="ಕನ್ನಡ ಸಾರಾಂಶ"
            ):
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

        with patch("summarizer.summarize_letter", return_value="ಕನ್ನಡ ಸಾರಾಂಶ"):
            app.button[0].click().run(timeout=15)

        self.assertIn("ಕನ್ನಡ ಸಾರಾಂಶ", [area.value for area in app.text_area])
        download_labels = [button.label for button in app.get("download_button")]
        self.assertIn("ಸಾರಾಂಶವನ್ನು UTF-8 ಪಠ್ಯವಾಗಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ", download_labels)

    def test_microphone_is_shown_after_summary_exists(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.session_state["ocr_result"] = {
            "text": "ಪತ್ರದ ಪಠ್ಯ",
            "words": [],
            "detected_languages": {},
        }
        app.session_state["summary_result"] = "ಕನ್ನಡ ಸಾರಾಂಶ"
        app.run(timeout=15)

        self.assertEqual(len(app.audio_input), 1)

    def test_summary_error_is_shown_in_streamlit(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.session_state["ocr_result"] = {
            "text": "ಪತ್ರದ ಪಠ್ಯ",
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


if __name__ == "__main__":
    unittest.main()
