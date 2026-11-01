import unittest
from unittest.mock import patch

import httpx

import tts


class TTSTests(unittest.TestCase):
    def test_request_uses_official_timbre_v3_format_and_returns_wav(self):
        response = httpx.Response(200, content=b"RIFFmock-wav")
        with (
            patch.dict("os.environ", {"GNANI_API_KEY": "test-key"}),
            patch("tts.httpx.post", return_value=response) as post,
        ):
            audio = tts.synthesize_speech(
                "ಕರ್ನಾಟಕ ಸರ್ಕಾರದಿಂದ ಬಂದಿದೆ.", voice="Shivani", speed=1.0
            )

        self.assertEqual(audio, b"RIFFmock-wav")
        post.assert_called_once_with(
            tts.TTS_ENDPOINT,
            headers={"X-API-Key-ID": "test-key"},
            json={
                "text": "ಕರ್ನಾಟಕ ಸರ್ಕಾರದಿಂದ ಬಂದಿದೆ.",
                "voice": "Shivani",
                "model": "timbre-v3.0",
                "language": "kn-IN",
                "speed": 1.0,
                "audio_config": {
                    "sample_rate": 48000,
                    "num_channels": 1,
                    "sample_width": 2,
                    "encoding": "linear_pcm",
                    "container": "wav",
                },
            },
            timeout=tts.TTS_TIMEOUT_SECONDS,
        )

    def test_missing_api_key_is_reported(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("tts.load_dotenv"):
                with self.assertRaisesRegex(tts.TTSError, "GNANI_API_KEY"):
                    tts.synthesize_speech("ಸಾರಾಂಶ")

    def test_api_error_is_reported(self):
        response = httpx.Response(
            401, json={"success": False, "message": "missing key"}
        )
        with (
            patch.dict("os.environ", {"GNANI_API_KEY": "bad-key"}),
            patch("tts.httpx.post", return_value=response),
        ):
            with self.assertRaises(tts.TTSError):
                tts.synthesize_speech("ಸಾರಾಂಶ")


if __name__ == "__main__":
    unittest.main()
