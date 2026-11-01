import io
import os
import wave
import array
import math
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


STT_ENDPOINT = "https://api.vachana.ai/stt/v3"
STT_TIMEOUT_SECONDS = 30


class STTError(Exception):
    """A safe, user-facing speech-to-text error."""


def inspect_wav_audio(audio_bytes: bytes) -> dict:
    if not audio_bytes:
        return {
            "byte_length": 0,
            "sample_rate": None,
            "channels": None,
            "sample_width": None,
            "duration": 0,
            "rms": 0,
            "has_signal": False,
        }

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    duration = frame_count / sample_rate if sample_rate else 0
    has_signal = False
    rms = 0.0
    if sample_width == 2 and len(frames) >= 2:
        samples = array.array("h")
        samples.frombytes(frames)
        if samples:
            if os.sys.byteorder == "big":
                samples.byteswap()
            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
            has_signal = any(sample != 0 for sample in samples)
    elif sample_width == 1 and frames:
        unsigned_samples = memoryview(frames).cast("B")
        rms = math.sqrt(
            sum((sample - 128) ** 2 for sample in unsigned_samples)
            / len(unsigned_samples)
        )
        has_signal = any(sample != 128 for sample in unsigned_samples)

    return {
        "byte_length": len(audio_bytes),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "duration": duration,
        "rms": rms,
        "has_signal": has_signal,
    }


def get_api_key() -> str:
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)

    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = st.secrets["GNANI_API_KEY"]
        except (KeyError, TypeError):
            api_key = ""

    api_key = str(api_key).strip()
    if not api_key or api_key == "paste_your_gnani_key_here":
        raise STTError(
            "Gnani API ಕೀ ಕಂಡುಬಂದಿಲ್ಲ. .env ಅಥವಾ Streamlit secrets‌ನಲ್ಲಿ "
            "GNANI_API_KEY ಹೊಂದಿಸಿ."
        )
    return api_key


def validate_wav_audio(audio_bytes: bytes) -> float:
    if not audio_bytes:
        raise STTError("ರೆಕಾರ್ಡಿಂಗ್ ಖಾಲಿಯಾಗಿದೆ. ಮತ್ತೆ ಮಾತಾಡಿ.")

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getframerate() != 16000:
                raise STTError("ಆಡಿಯೊ 16000 Hz ನಲ್ಲಿ ಇರಬೇಕು.")
            if wav_file.getnchannels() not in (1, 2):
                raise STTError("The recording has an unsupported number of channels.")
            if wav_file.getsampwidth() not in (1, 2):
                raise STTError("The recording has an unsupported WAV sample format.")
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration = frames / frame_rate if frame_rate else 0
    except STTError:
        raise
    except (EOFError, OSError, wave.Error):
        raise STTError("ಮಾನ್ಯವಾದ WAV ರೆಕಾರ್ಡಿಂಗ್ ಕಂಡುಬಂದಿಲ್ಲ.") from None

    if duration <= 0:
        raise STTError("ರೆಕಾರ್ಡಿಂಗ್ ಖಾಲಿಯಾಗಿದೆ. ಮತ್ತೆ ಮಾತಾಡಿ.")
    if duration > 30:
        raise STTError("ರೆಕಾರ್ಡಿಂಗ್ 30 ಸೆಕೆಂಡುಗಳಿಗಿಂತ ಕಡಿಮೆ ಇರಬೇಕು.")
    if not inspect_wav_audio(audio_bytes)["has_signal"]:
        raise STTError(
            "The recording contains only silence. Check browser microphone permission, "
            "selected microphone and operating-system privacy settings, then record again."
        )
    return duration


def transcribe_kannada(audio_bytes: bytes) -> tuple[str, str]:
    validate_wav_audio(audio_bytes)
    api_key = get_api_key()

    try:
        response = requests.post(
            STT_ENDPOINT,
            headers={"X-API-Key-ID": api_key},
            data={
                "model": "prisma-v3.0",
                "language_code": "kn-IN",
                "format": "transcribe",
            },
            files={"audio_file": ("recording.wav", audio_bytes, "audio/wav")},
            timeout=STT_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        raise STTError(
            "Gnani API ಸಂಪರ್ಕ ಸಮಯ ಮೀರಿದೆ. ನೆಟ್‌ವರ್ಕ್ ಪರಿಶೀಲಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        ) from None
    except requests.RequestException:
        raise STTError(
            "Gnani API ಸಂಪರ್ಕ ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ನೆಟ್‌ವರ್ಕ್ ಪರಿಶೀಲಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        ) from None

    if response.status_code in (401, 403):
        raise STTError(
            "Authentication failed. Gnani API key and STT permission should be checked."
        )
    if response.status_code == 402:
        raise STTError(
            "Insufficient Gnani credits. Please add credits and try again."
        )
    if response.status_code == 429:
        raise STTError(
            "Gnani STT request limit reached. Please wait and try again."
        )
    if response.status_code in (400, 422) or response.status_code >= 500:
        if response.status_code >= 500:
            raise STTError(
                "Gnani speech service is temporarily unavailable. Please try again."
            )
        raise STTError(
            "The recorded audio could not be processed. Please record it again."
        )
    if response.status_code >= 400:
        raise STTError(
            f"Gnani STT request failed (HTTP {response.status_code})."
        )

    try:
        result = response.json()
    except ValueError:
        raise STTError("Gnani API returned an unexpected response.") from None

    if not isinstance(result, dict):
        raise STTError("Gnani API returned an unexpected response.")

    transcript = str(result.get("transcript", "")).strip()
    detected_language = str(result.get("detected_language_code", "")).strip()
    if not transcript:
        raise STTError("No speech transcription was returned. Please try another recording.")
    if not detected_language:
        raise STTError("Gnani API did not return the detected language code.")
    return transcript, detected_language
