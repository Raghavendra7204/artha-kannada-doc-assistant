import io
import unittest
import wave
from unittest.mock import patch

import stt


def wav_bytes(duration=0.1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\0\0" * int(16000 * duration))
    return buffer.getvalue()


def wav_bytes_with_samples(duration=0.1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes((1000).to_bytes(2, "little", signed=True) * int(16000 * duration))
    return buffer.getvalue()


class STTTests(unittest.TestCase):
    def test_transcribe_posts_required_fields_and_returns_result(self):
        response = unittest.mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "transcript": "ನಮಸ್ಕಾರ",
            "detected_language_code": "kn-IN",
        }

        with (
            patch("stt.get_api_key", return_value="test-key"),
            patch("stt.requests.post", return_value=response) as post,
        ):
            result = stt.transcribe_kannada(wav_bytes_with_samples())

        self.assertEqual(result, ("ನಮಸ್ಕಾರ", "kn-IN"))
        args, kwargs = post.call_args
        self.assertEqual(args[0], stt.STT_ENDPOINT)
        self.assertEqual(kwargs["headers"], {"X-API-Key-ID": "test-key"})
        self.assertEqual(
            kwargs["data"],
            {
                "model": "prisma-v3.0",
                "language_code": "kn-IN",
                "format": "transcribe",
            },
        )
        self.assertEqual(kwargs["files"]["audio_file"][2], "audio/wav")

    def test_empty_transcript_is_reported(self):
        response = unittest.mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "transcript": "",
            "detected_language_code": "kn-IN",
        }

        with (
            patch("stt.get_api_key", return_value="test-key"),
            patch("stt.requests.post", return_value=response),
        ):
            with self.assertRaisesRegex(stt.STTError, "No speech transcription"):
                stt.transcribe_kannada(wav_bytes_with_samples())

    def test_silent_wav_is_rejected(self):
        with self.assertRaisesRegex(stt.STTError, "silence"):
            stt.validate_wav_audio(wav_bytes())


if __name__ == "__main__":
    unittest.main()
