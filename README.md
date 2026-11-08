# Artha

## Government Letters, Made Simple.

Artha is a Kannada-first Streamlit assistant for understanding government
letters. It extracts text from document images and PDFs, explains the document
in simple Kannada, reads the explanation aloud, and answers spoken questions
using only the current document text.

## Why Artha

Government letters often contain formal language, dates, deadlines, reference
numbers, and required actions that are difficult to understand quickly. Artha
helps a Kannada-speaking reader access that information through text, speech,
and document-grounded questions while keeping the original extracted text
available for verification.

## How it works

```text
Document Upload
      ↓
OCR / Embedded Text Extraction
      ↓
Kannada Summarization
      ↓
Text-to-Speech
      ↓
Voice Question
      ↓
Speech-to-Text
      ↓
Document-Grounded Answer
      ↓
Optional Spoken Response
```

Summary and answer text are generated only from the current document text.
The original extracted text is kept separate from speech-oriented text
normalization used for Timbre audio.

## Features

- JPG, JPEG, PNG, and multi-page PDF document support.
- Direct text extraction for suitable PDF pages.
- Google Cloud Vision OCR fallback for scanned PDF pages and images.
- Page-order-preserving combined text with page labels for PDFs.
- Kannada summary generation with Gemini.
- Downloadable OCR text and Kannada summary.
- Kannada summary playback through Gnani AI Timbre.
- Kannada voice questions through Gnani AI Prisma.
- Document-grounded follow-up answers with optional spoken playback.
- Background prefetching for predictable summary and audio operations without
  automatically revealing or playing results.

## Technology stack

- Python
- Streamlit
- Pillow
- PyMuPDF
- Google Cloud Vision REST API
- Google GenAI SDK for Gemini
- Gnani AI Prisma STT and Timbre TTS APIs
- `requests` and `httpx` for external API calls

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point and user interaction flow |
| `ocr.py` | Image preparation, Google Vision OCR, and PDF processing |
| `summarizer.py` | Gemini summary and document-grounded Q&A calls |
| `tts.py` | Gnani Timbre text-to-speech integration |
| `stt.py` | Gnani Prisma speech-to-text integration and audio validation |
| `tts_normalizer.py` | Deterministic speech-only normalization for numbers, dates, times, identifiers, and known abbreviations |
| `background_tasks.py` | Bounded background task registry for safe prefetching |
| `assets/artha_logo.png` | Application logo |
| `tests/` | Unit and Streamlit application tests |

## Setup

Create and activate a virtual environment, then install the existing
dependencies:

```text
python -m venv .venv
```

Windows PowerShell:

```text
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the environment template and fill in your own credentials:

```text
Copy-Item .env.example .env
```

Never commit `.env`. It is ignored by Git.

## Environment variables

Artha reads these variables from `.env`:

```text
GOOGLE_VISION_API_KEY=your_google_cloud_vision_key
GEMINI_API_KEY=your_gemini_api_key
GNANI_API_KEY=your_gnani_api_key
```

The Google Cloud Vision API must be enabled for the Google project associated
with the Vision key. API access, quotas, billing, and model availability are
controlled by the respective providers.

## Run the application

```text
streamlit run app.py
```

Upload one image or PDF, or capture a document photo. Click the text-reading
action, review the extracted text if needed, and request the Kannada summary.
After the summary is available, record a Kannada question and request its
transcription and answer.

## Supported documents

- Images: `.jpg`, `.jpeg`, `.png`
- PDFs: multi-page PDFs processed in original page order
- PDF pages with usable embedded text use direct extraction.
- Scanned or image-heavy PDF pages use the Google Vision OCR pipeline.
- A PDF is limited to 50 MB by the current processing safeguards.
- Failed or unreadable pages produce page-specific warnings and are not
  silently presented as successfully extracted text.

## Limitations and privacy considerations

- Google Vision, Gemini, and Gnani API credentials are required for their
  respective operations.
- Document text and voice-related data are sent to external provider APIs when
  those operations are requested. Do not upload sensitive documents unless
  that use is acceptable for your environment and provider agreements.
- Provider quotas, network availability, request limits, and Gemini input-size
  limits can affect processing.
- OCR quality depends on image clarity, document layout, language, and scan
  quality.
- The summary and answers are reading aids, not legal verification or legal
  advice.
- Background prefetch state is process-local and is not persisted across an
  application restart. Results are still revealed or played only after the
  corresponding user action.

## Acknowledgments

Artha uses Google Cloud Vision and Gemini, and integrates Gnani AI Prisma for
speech-to-text and Gnani AI Timbre for text-to-speech.

## Tests

Run the automated tests with:

```text
python -m unittest discover -s tests -p "test_*.py"
```

The tests mock external API calls and cover OCR/PDF behavior, summarization,
TTS/STT integrations, speech normalization, Streamlit flows, and background
task deduplication/retry behavior.
