import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


TTS_ENDPOINT = "https://api.vachana.ai/api/v1/tts/inference"
TTS_MODEL = "timbre-v3.0"
TTS_LANGUAGE = "kn-IN"
TTS_TIMEOUT_SECONDS = 30

TIMBRE_V3_VOICES = (
    "Shivani",
    "Lakshmi",
    "Kalpana",
    "Chetana",
    "Vasudha",
    "Riyaa",
    "Tarika",
    "Avinash",
    "Shubham",
    "Prakash",
    "Devansh",
    "Yashraj",
    "Harshad",
)


class TTSError(Exception):
    """A safe, user-facing text-to-speech error."""


def synthesize_speech(text: str, voice: str = "Shivani", speed: float = 1.0) -> bytes:
    load_dotenv(
        dotenv_path=Path(__file__).resolve().with_name(".env"), override=False
    )
    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if not api_key or api_key == "paste_your_gnani_key_here":
        raise TTSError(
            "GNANI_API_KEY Gnani API ಕೀ ಕಂಡುಬಂದಿಲ್ಲ. .env ಫೈಲ್‌ನಲ್ಲಿ GNANI_API_KEY ಹೊಂದಿಸಿ."
        )

    if not text or not text.strip():
        raise TTSError("ಧ್ವನಿ ಉತ್ಪಾದಿಸಲು ಕನ್ನಡ ಸಾರಾಂಶದ ಪಠ್ಯ ಲಭ್ಯವಿಲ್ಲ.")

    payload = {
        "text": text,
        "voice": voice,
        "model": TTS_MODEL,
        "language": TTS_LANGUAGE,
        "speed": speed,
        "audio_config": {
            "sample_rate": 48000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }

    try:
        response = httpx.post(
            TTS_ENDPOINT,
            headers={"X-API-Key-ID": api_key},
            json=payload,
            timeout=TTS_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException:
        raise TTSError(
            "Gnani API ಸಂಪರ್ಕ ಸಮಯ ಮೀರಿದೆ. ನೆಟ್‌ವರ್ಕ್ ಪರಿಶೀಲಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        ) from None
    except httpx.HTTPError:
        raise TTSError(
            "Gnani API ಸಂಪರ್ಕ ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ನೆಟ್‌ವರ್ಕ್ ಪರಿಶೀಲಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        ) from None

    if response.status_code >= 400:
        try:
            message = response.json().get("message")
        except (ValueError, AttributeError):
            message = None
        status = response.status_code
        if status in (401, 403):
            detail = "GNANI_API_KEY ಮತ್ತು Gnani API ಪ್ರವೇಶವನ್ನು ಪರಿಶೀಲಿಸಿ."
        elif status == 429:
            detail = "Gnani API ಬಳಕೆಯ ಮಿತಿ ತಲುಪಿದೆ. ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        elif status >= 500:
            detail = "Gnani ಸೇವೆಯಲ್ಲಿ ತಾತ್ಕಾಲಿಕ ದೋಷವಿದೆ. ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        else:
            detail = message or "ವಿನಂತಿ ಮಾನ್ಯವಾಗಿಲ್ಲ."
        raise TTSError(f"ಧ್ವನಿ ಉತ್ಪಾದನೆ ವಿಫಲವಾಗಿದೆ: {detail} (HTTP {status})")

    if not response.content:
        raise TTSError("Gnani API ಯಾವುದೇ ಆಡಿಯೊವನ್ನು ಹಿಂತಿರುಗಿಸಲಿಲ್ಲ.")
    return response.content
