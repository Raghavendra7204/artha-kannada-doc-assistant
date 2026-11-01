import hashlib

import streamlit as st

from ocr import prepare_image, run_ocr
from summarizer import (
    AnswerQuestionError,
    SummarizationError,
    answer_letter_question,
    summarize_letter,
)
from stt import STTError, inspect_wav_audio, transcribe_kannada, validate_wav_audio
from tts import TIMBRE_V3_VOICES, TTSError, synthesize_speech


st.set_page_config(page_title="Samaj – Kannada Government Letter Assistant")
st.title("Samaj – Kannada Government Letter Assistant")

uploaded_file = st.file_uploader(
    "Upload a document photo", type=["jpg", "jpeg", "png"]
)
with st.expander("Or take a photo"):
    camera_file = st.camera_input("Take a document photo")

image_file = uploaded_file or camera_file
if image_file is not None:
    image_bytes = image_file.getvalue()
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.get("ocr_image_digest") != image_digest:
        for key in (
            "ocr_result",
            "ocr_error",
            "summary_result",
            "summary_error",
            "summary_audio",
            "summary_audio_key",
            "conversation",
            "question_transcript",
            "question_error",
            "answer_error",
            "answer_audio",
            "qa_ocr_text",
        ):
            st.session_state.pop(key, None)
        st.session_state["ocr_image_digest"] = image_digest

    st.image(image_bytes, caption="Document preview", use_container_width=True)
    if st.button("Read text"):
        with st.spinner("Reading the document..."):
            try:
                st.session_state["ocr_result"] = run_ocr(prepare_image(image_bytes))
                st.session_state.pop("ocr_error", None)
                st.session_state.pop("summary_result", None)
                st.session_state.pop("summary_error", None)
                st.session_state.pop("summary_audio", None)
                st.session_state.pop("summary_audio_key", None)
                st.session_state["conversation"] = []
                st.session_state.pop("question_transcript", None)
                st.session_state.pop("question_error", None)
                st.session_state.pop("answer_error", None)
                st.session_state.pop("answer_audio", None)
                st.session_state["qa_ocr_text"] = st.session_state["ocr_result"]["text"]
            except Exception as error:
                st.session_state["ocr_error"] = str(error)
                st.session_state.pop("ocr_result", None)
                st.session_state.pop("summary_result", None)
                st.session_state.pop("summary_error", None)
                st.session_state.pop("summary_audio", None)
                st.session_state.pop("summary_audio_key", None)
                st.session_state["conversation"] = []
                st.session_state.pop("question_transcript", None)
                st.session_state.pop("question_error", None)
                st.session_state.pop("answer_error", None)
                st.session_state.pop("answer_audio", None)

if "ocr_error" in st.session_state:
    st.error(st.session_state["ocr_error"])

if "ocr_result" in st.session_state:
    result = st.session_state["ocr_result"]
    st.text_area("Extracted text", value=result["text"], height=400, disabled=True)
    words = result["words"]
    if words:
        mean_confidence = sum(word["confidence"] for word in words) / len(words)
        st.metric("Mean word confidence", f"{mean_confidence:.1%}")
    else:
        st.metric("Mean word confidence", "N/A")
    st.write("Detected languages", result["detected_languages"] or "None")
    lowest_confidence = sorted(words, key=lambda word: word["confidence"])[:10]
    st.write(
        "Lowest-confidence words",
        [
            {"word": word["text"], "confidence": word["confidence"]}
            for word in lowest_confidence
        ],
    )
    st.download_button(
        "Download text",
        data=result["text"].encode("utf-8"),
        file_name="kannada_ocr.txt",
        mime="text/plain",
    )

    if not result["text"].strip():
        st.info(
            "OCR ಮೂಲಕ ಬಳಕೆ ಮಾಡಬಹುದಾದ ಪಠ್ಯ ದೊರಕಿಲ್ಲ. "
            "ಸಾರಾಂಶ ರಚಿಸಲು ಮೊದಲು ಮತ್ತೊಮ್ಮೆ OCR ಪ್ರಯತ್ನಿಸಿ."
        )
    elif st.button("ಸಾರಾಂಶ ರಚಿಸಿ"):
        with st.spinner("ಪತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ ಸಾರಾಂಶ ರಚಿಸಲಾಗುತ್ತಿದೆ..."):
            try:
                st.session_state["summary_result"] = summarize_letter(result["text"])
                st.session_state.pop("summary_error", None)
                st.session_state.pop("summary_audio", None)
                st.session_state.pop("summary_audio_key", None)
            except SummarizationError as error:
                st.session_state["summary_error"] = str(error)
                st.session_state.pop("summary_result", None)

if "summary_error" in st.session_state:
    st.error(st.session_state["summary_error"])

if "summary_result" in st.session_state:
    summary = st.session_state["summary_result"]
    st.subheader("ಕನ್ನಡ ಸಾರಾಂಶ ಮತ್ತು ಪ್ರಮುಖ ಮಾಹಿತಿ")
    st.text_area("ಸಾರಾಂಶ", value=summary, height=500, disabled=True)
    st.download_button(
        "ಸಾರಾಂಶವನ್ನು UTF-8 ಪಠ್ಯವಾಗಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        data=summary.encode("utf-8"),
        file_name="kannada_letter_summary.txt",
        mime="text/plain; charset=utf-8",
    )

    voice = st.selectbox("ಧ್ವನಿ", TIMBRE_V3_VOICES, index=0)
    speed = st.slider("ಚಲನೆಯ ವೇಗ", min_value=0.85, max_value=1.15, value=1.0, step=0.05)
    audio_key = (summary, voice, speed)

    if st.button("ಕೇಳಿ"):
        with st.spinner("ಆಡಿಯೊ ತಯಾರಿಸಲಾಗುತ್ತಿದೆ..."):
            try:
                audio = synthesize_speech(summary, voice=voice, speed=speed)
                st.session_state["summary_audio"] = audio
                st.session_state["summary_audio_key"] = audio_key
            except TTSError as error:
                st.error(str(error))

    if (
        st.session_state.get("summary_audio_key") == audio_key
        and st.session_state.get("summary_audio")
    ):
        st.audio(st.session_state["summary_audio"], format="audio/wav")

    current_ocr_text = st.session_state["ocr_result"]["text"]
    if st.session_state.get("qa_ocr_text") != current_ocr_text:
        st.session_state["conversation"] = []
        st.session_state["qa_ocr_text"] = current_ocr_text
        st.session_state.pop("question_transcript", None)
        st.session_state.pop("answer_audio", None)

    st.subheader("ಪತ್ರದ ಬಗ್ಗೆ ಕೇಳಿ")
    recorded_audio = st.audio_input(
        "Kannada question recording",
        sample_rate=16000,
        help="Record up to 30 seconds, then press Stop.",
    )
    if recorded_audio is not None:
        audio_bytes = recorded_audio.getvalue()
        if not audio_bytes:
            st.error("ರೆಕಾರ್ಡಿಂಗ್ ಖಾಲಿಯಾಗಿದೆ. ಮತ್ತೆ ಮಾತಾಡಿ.")
        else:
            st.audio(audio_bytes, format="audio/wav")
            try:
                validate_wav_audio(audio_bytes)
                valid_audio = True
            except STTError as error:
                st.error(str(error))
                valid_audio = False
            diagnostics = inspect_wav_audio(audio_bytes)
            with st.expander("Recording diagnostics"):
                st.write({
                    "Bytes": diagnostics["byte_length"],
                    "MIME type": getattr(recorded_audio, "type", None) or "unknown",
                    "Sample rate": diagnostics["sample_rate"],
                    "Channels": diagnostics["channels"],
                    "Sample width (bytes)": diagnostics["sample_width"],
                    "Duration (seconds)": round(diagnostics["duration"], 2),
                    "PCM RMS": round(diagnostics["rms"], 2),
                    "Contains non-zero samples": diagnostics["has_signal"],
                })

            if valid_audio and st.button("ಪ್ರಶ್ನೆಯನ್ನು ಗುರುತಿಸಿ"):
                st.session_state.pop("answer_error", None)
                with st.spinner("Kannada question transcribing..."):
                    try:
                        transcript, detected_language = transcribe_kannada(audio_bytes)
                        transcript = transcript.strip()
                        if not transcript:
                            raise STTError(
                                "ಪ್ರಶ್ನೆಯನ್ನು ಗುರುತಿಸಲಾಗಿಲ್ಲ. ಮತ್ತೆ ಮಾತಾಡಿ."
                            )
                        st.session_state["question_transcript"] = transcript
                        with st.spinner("ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲಾಗುತ್ತಿದೆ..."):
                            answer = answer_letter_question(
                                current_ocr_text, transcript
                            )
                        st.session_state.setdefault("conversation", []).append(
                            {"question": transcript, "answer": answer}
                        )
                        st.session_state.pop("question_error", None)
                    except STTError as error:
                        st.session_state["question_error"] = str(error)
                    except AnswerQuestionError as error:
                        st.session_state["answer_error"] = str(error)

    if st.session_state.get("question_error"):
        st.error(st.session_state["question_error"])
    if st.session_state.get("answer_error"):
        st.error(st.session_state["answer_error"])
    if st.session_state.get("question_transcript"):
        st.subheader("ಗುರುತಿಸಿದ ಪ್ರಶ್ನೆ")
        st.write(st.session_state["question_transcript"])

    conversation = st.session_state.setdefault("conversation", [])
    if conversation:
        st.subheader("ಪ್ರಶ್ನೆ ಮತ್ತು ಉತ್ತರಗಳು")
        for index, item in enumerate(conversation):
            st.markdown(f"**ಪ್ರಶ್ನೆ:** {item['question']}")
            st.markdown(f"**ಉತ್ತರ:** {item['answer']}")

            voice = st.selectbox(
                "ಉತ್ತರದ ಧ್ವನಿ",
                TIMBRE_V3_VOICES,
                index=0,
                key=f"answer_voice_{index}",
            )
            speed = st.slider(
                "ಉತ್ತರದ ಚಲನೆಯ ವೇಗ",
                min_value=0.85,
                max_value=1.15,
                value=1.0,
                step=0.05,
                key=f"answer_speed_{index}",
            )
            audio_key = (index, item["answer"], voice, speed)
            if st.button("ಉತ್ತರ ಕೇಳಿ", key=f"answer_tts_{index}"):
                try:
                    st.session_state.setdefault("answer_audio", {})[audio_key] = (
                        synthesize_speech(item["answer"], voice=voice, speed=speed)
                    )
                except TTSError as error:
                    st.error(str(error))
            if st.session_state.get("answer_audio", {}).get(audio_key):
                st.audio(
                    st.session_state["answer_audio"][audio_key],
                    format="audio/wav",
                )
