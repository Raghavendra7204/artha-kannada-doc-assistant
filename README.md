# Samaj – Kannada Government Letter Assistant

Upload or photograph a Kannada government letter, extract its text with the
existing Google Cloud Vision OCR workflow, and optionally generate a structured
Kannada summary with the Gemini API. OCR and summarization are separate actions;
the Gemini API is called only when you click **ಸಾರಾಂಶ ರಚಿಸಿ**.

## Setup

1. If `.env` does not already exist, copy `.env.example` to `.env`. If you
   already have a `.env` file, keep it and add or update the Gemini setting in
   that file; do not overwrite your existing Cloud Vision key.
2. Set `GOOGLE_VISION_API_KEY` to a Google Cloud Vision API key. The existing
   OCR feature requires a Google Cloud project with the Cloud Vision API enabled
   and billing configured; restrict this key to Cloud Vision API.
3. Set `GEMINI_API_KEY` to a key from
   [Google AI Studio](https://aistudio.google.com/apikey). Keep `.env` private;
   it is excluded by `.gitignore`.
4. Create and activate a virtual environment, then install dependencies:
   ```text
   python -m venv .venv
   ```
   Activate `.venv` for your operating system, then run:
   ```text
   pip install -r requirements.txt
   ```
5. Start the existing Streamlit app:
   ```text
   streamlit run app.py
   ```

Upload or take a document photo and click **Read text** to run OCR. Review the
extracted text, then click **ಸಾರಾಂಶ ರಚಿಸಿ** to request the Kannada summary.
Both the OCR text and summary can be downloaded as UTF-8 text files. The model
is configured by `GEMINI_MODEL` in `summarizer.py`.

After the summary is shown, the Kannada microphone becomes available. Record a
question, stop the recording, and click **ಪ್ರಶ್ನೆಯನ್ನು ಗುರುತಿಸಿ** to transcribe
it with Gnani Prisma v3.0 and ask Gemini about the current letter. Gemini
answers only from the OCR text. Previous answers remain visible for the same
letter, and a newly uploaded letter clears the conversation.

The application currently uses Google's `gemini-3.6-flash` model. Free-tier
model access, quotas, and availability depend on Google's current Gemini API
terms and your account eligibility; a free API tier is not guaranteed for every
model or account. Check Google's current
[pricing](https://ai.google.dev/gemini-api/docs/pricing) and
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits). OCR text is
sent to Google when you request a summary; review the
[Gemini API terms](https://ai.google.dev/gemini-api/terms), especially before
sending sensitive letters. The summary is an aid for reading and is not legally
verified.

OCR can also be run from a file with `python ocr.py samples/x.jpg`.

On Windows, the terminal may garble Kannada; use the app or open the UTF-8 output file in `out/`.
