import hashlib
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from background_tasks import make_task_key, submit_task
from ocr import pdf_page_count, prepare_image, process_pdf, run_ocr
from summarizer import (
    AnswerQuestionError,
    GEMINI_MODEL,
    QUESTION_SYSTEM_PROMPT,
    SummarizationError,
    SYSTEM_PROMPT,
    answer_letter_question,
    summarize_letter,
)
from stt import STTError, inspect_wav_audio, transcribe_kannada, validate_wav_audio
from tts import (
    TTS_ENDPOINT,
    TTS_LANGUAGE,
    TTS_MODEL,
    TIMBRE_V3_VOICES,
    TTSError,
    synthesize_speech,
)
from tts_normalizer import normalize_for_tts


TTS_CONFIG_KEY = (TTS_ENDPOINT, TTS_MODEL, TTS_LANGUAGE, 48000, 1, 2, "linear_pcm", "wav")


def summary_task_key(ocr_text):
    return make_task_key("summary", GEMINI_MODEL, SYSTEM_PROMPT, ocr_text)


def summary_audio_task_key(document_text, text, voice, speed):
    return make_task_key(
        "summary-audio",
        GEMINI_MODEL,
        document_text,
        text,
        voice,
        speed,
        TTS_CONFIG_KEY,
    )


def answer_audio_task_key(document_text, answer, voice, speed):
    return make_task_key(
        "answer-audio",
        GEMINI_MODEL,
        QUESTION_SYSTEM_PROMPT,
        document_text,
        answer,
        voice,
        speed,
        TTS_CONFIG_KEY,
    )


st.set_page_config(page_title="Artha – Kannada Government Letter Assistant")
logo_path = Path(__file__).resolve().parent / "assets" / "artha_logo.png"
st.image(str(logo_path), width=600)
st.subheader("ಪತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ")

uploaded_file = st.file_uploader(
    "ಪತ್ರದ ಚಿತ್ರ ಅಥವಾ PDF ಆಯ್ಕೆಮಾಡಿ", type=["jpg", "jpeg", "png", "pdf"]
)
with st.expander("ಅಥವಾ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ"):
    camera_file = st.camera_input("ಪತ್ರದ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ")

image_file = uploaded_file or camera_file
if image_file is not None:
    image_bytes = image_file.getvalue()
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    is_pdf = (
        getattr(image_file, "type", "") == "application/pdf"
        or str(getattr(image_file, "name", "")).lower().endswith(".pdf")
    )
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

    if is_pdf:
        try:
            page_count = pdf_page_count(image_bytes)
            st.write(f"PDF: {getattr(image_file, 'name', 'document.pdf')} ({page_count} pages)")
        except ValueError as error:
            st.warning(str(error))
    else:
        preview = Image.open(BytesIO(image_bytes))
        preview.thumbnail((900, 340))
        st.image(preview, caption="ಪತ್ರದ ಪೂರ್ವವೀಕ್ಷಣೆ", width="content")
    if st.button("ಪಠ್ಯ ಓದಿ", icon=":material/document_scanner:"):
        progress = st.progress(0, text="Processing document...") if is_pdf else None
        with st.spinner("Reading the document..."):
            try:
                if is_pdf:
                    st.session_state["ocr_result"] = process_pdf(
                        image_bytes,
                        progress_callback=lambda page, total: progress.progress(
                            page / total, text=f"Processing page {page} of {total}"
                        ),
                    )
                else:
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
                ocr_text = st.session_state["ocr_result"]["text"]
                submit_task(
                    summary_task_key(ocr_text), summarize_letter, ocr_text
                )
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
            finally:
                if progress is not None:
                    progress.empty()

if "ocr_error" in st.session_state:
    st.error(st.session_state["ocr_error"])

if "ocr_result" in st.session_state:
    result = st.session_state["ocr_result"]
    with st.expander("ಹೊರತೆಗೆದ ಪಠ್ಯವನ್ನು ನೋಡಿ"):
        st.text_area("Extracted text", value=result["text"], height=400, disabled=True)
    for warning in result.get("page_warnings", []):
        st.warning(warning)
    st.download_button(
        "ಪಠ್ಯ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        data=result["text"].encode("utf-8"),
        file_name="kannada_ocr.txt",
        mime="text/plain",
        icon=":material/download:",
    )

    if not result["text"].strip():
        st.info(
            "OCR ಮೂಲಕ ಬಳಕೆ ಮಾಡಬಹುದಾದ ಪಠ್ಯ ದೊರಕಿಲ್ಲ. "
            "ಸಾರಾಂಶ ರಚಿಸಲು ಮೊದಲು ಮತ್ತೊಮ್ಮೆ OCR ಪ್ರಯತ್ನಿಸಿ."
        )
    elif st.button("ಸಾರಾಂಶ ರಚಿಸಿ", icon=":material/description:"):
        with st.spinner("ಪತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ ಸಾರಾಂಶ ರಚಿಸಲಾಗುತ್ತಿದೆ..."):
            try:
                summary_key = summary_task_key(result["text"])
                summary_task = submit_task(
                    summary_key,
                    summarize_letter,
                    result["text"],
                    retry_failed=True,
                )
                st.session_state["summary_result"] = (
                    summary_task.result()
                    if summary_task is not None
                    else summarize_letter(result["text"])
                )
                st.session_state.pop("summary_error", None)
                st.session_state.pop("summary_audio", None)
                st.session_state.pop("summary_audio_key", None)
            except SummarizationError as error:
                st.session_state["summary_error"] = str(error)
                st.session_state.pop("summary_result", None)
            except Exception as error:
                st.session_state["summary_error"] = str(error)
                st.session_state.pop("summary_result", None)

if "summary_error" in st.session_state:
    st.error(st.session_state["summary_error"])

if "summary_result" in st.session_state:
    summary = st.session_state["summary_result"]
    st.divider()
    st.subheader("ಪತ್ರದ ಸಾರಾಂಶ")
    st.write(summary)
    st.download_button(
        "ಸಾರಾಂಶ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        data=summary.encode("utf-8"),
        file_name="kannada_letter_summary.txt",
        mime="text/plain; charset=utf-8",
        icon=":material/download:",
    )

    with st.expander("ಧ್ವನಿ ಸೆಟ್ಟಿಂಗ್‌ಗಳು"):
        voice = st.selectbox("ಧ್ವನಿ", TIMBRE_V3_VOICES, index=0)
        speed = st.slider("ಚಲನೆಯ ವೇಗ", min_value=0.85, max_value=1.15, value=1.0, step=0.05)
    tts_summary = normalize_for_tts(summary)
    audio_key = summary_audio_task_key(result["text"], tts_summary, voice, speed)
    summary_audio_task = submit_task(
        audio_key,
        synthesize_speech,
        tts_summary,
        voice=voice,
        speed=speed,
    )

    if st.button("ಕೇಳಿ", icon=":material/volume_up:"):
        with st.spinner("ಆಡಿಯೊ ತಯಾರಿಸಲಾಗುತ್ತಿದೆ..."):
            try:
                summary_audio_task = submit_task(
                    audio_key,
                    synthesize_speech,
                    tts_summary,
                    voice=voice,
                    speed=speed,
                    retry_failed=True,
                )
                audio = (
                    summary_audio_task.result()
                    if summary_audio_task is not None
                    else synthesize_speech(tts_summary, voice=voice, speed=speed)
                )
                st.session_state["summary_audio"] = audio
                st.session_state["summary_audio_key"] = audio_key
            except TTSError as error:
                st.error(str(error))
            except Exception as error:
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

    st.divider()
    st.subheader("ಪತ್ರದ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ")
    recorded_audio = st.audio_input(
        "ಮೈಕ್ ಒತ್ತಿ ಕನ್ನಡದಲ್ಲಿ ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಹೇಳಿ.",
        sample_rate=16000,
        help="30 ಸೆಕೆಂಡುಗಳವರೆಗೆ ರೆಕಾರ್ಡ್ ಮಾಡಿ, ನಂತರ ನಿಲ್ಲಿಸಿ.",
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

            with st.expander("ಧ್ವನಿ ಸೆಟ್ಟಿಂಗ್‌ಗಳು"):
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
            tts_answer = normalize_for_tts(item["answer"])
            audio_key = answer_audio_task_key(
                current_ocr_text, tts_answer, voice, speed
            )
            submit_task(
                audio_key,
                synthesize_speech,
                tts_answer,
                voice=voice,
                speed=speed,
            )
            if st.button("ಉತ್ತರ ಕೇಳಿ", key=f"answer_tts_{index}"):
                try:
                    answer_audio_task = submit_task(
                        audio_key,
                        synthesize_speech,
                        tts_answer,
                        voice=voice,
                        speed=speed,
                        retry_failed=True,
                    )
                    st.session_state.setdefault("answer_audio", {})[audio_key] = (
                        answer_audio_task.result()
                        if answer_audio_task is not None
                        else synthesize_speech(
                            tts_answer, voice=voice, speed=speed
                        )
                    )
                except TTSError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(str(error))
            if st.session_state.get("answer_audio", {}).get(audio_key):
                st.audio(
                    st.session_state["answer_audio"][audio_key],
                    format="audio/wav",
                )
