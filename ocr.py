import argparse
import base64
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
import pymupdf as fitz
from PIL import Image, ImageOps


VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
PLACEHOLDER_KEY = "paste_your_key_here"
MAX_PDF_BYTES = 50 * 1024 * 1024
RASTER_COVERAGE_THRESHOLD = 0.25


def prepare_image(image_bytes):
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max(image.size) > 2000:
            image.thumbnail((2000, 2000))
        output = BytesIO()
        image.save(output, format="JPEG", quality=90)
        return output.getvalue()


def _api_key():
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=True)
    key = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
    if not key or key == PLACEHOLDER_KEY:
        raise RuntimeError(
            "Google Cloud Vision API key is missing or still the placeholder. "
            "Set GOOGLE_VISION_API_KEY in your .env file."
        )
    return key


def _confidence(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def run_ocr(image_bytes, hints=("kn", "en")):
    key = _api_key()
    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": list(hints)},
            }
        ]
    }
    try:
        response = requests.post(
            VISION_URL, params={"key": key}, json=payload, timeout=60
        )
    except requests.RequestException:
        raise RuntimeError(
            "Could not reach Google Cloud Vision. Check your network and try again."
        ) from None

    if response.status_code == 400:
        raise RuntimeError("Google Cloud Vision rejected the request (400): bad image or request.")
    if response.status_code == 403:
        raise RuntimeError(
            "Google Cloud Vision access was denied (403). Check that the API key is valid, "
            "restricted to Cloud Vision API, and that the API and billing are enabled."
        )
    if response.status_code == 429:
        raise RuntimeError("Google Cloud Vision quota exceeded (429). Try again later.")
    if not response.ok:
        raise RuntimeError(
            f"Google Cloud Vision returned HTTP {response.status_code}."
        )

    try:
        data = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        raise RuntimeError("Google Cloud Vision returned an invalid response.") from None

    responses = data.get("responses", []) if isinstance(data, dict) else []
    result = (
        responses[0]
        if isinstance(responses, list) and responses and isinstance(responses[0], dict)
        else {}
    )
    error = result.get("error", {})
    if isinstance(error, dict) and error.get("message"):
        message = str(error["message"]).replace(key, "[redacted]")
        raise RuntimeError(f"Google Cloud Vision error: {message}")

    annotation = result.get("fullTextAnnotation", {})
    text = annotation.get("text", "") if isinstance(annotation, dict) else ""
    if not isinstance(text, str):
        text = ""

    words = []
    detected_languages = {}
    pages = annotation.get("pages", []) if isinstance(annotation, dict) else []
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            prop = page.get("property", {})
            languages = prop.get("detectedLanguages", []) if isinstance(prop, dict) else []
            if isinstance(languages, list):
                for language in languages:
                    if isinstance(language, dict) and language.get("languageCode"):
                        detected_languages[str(language["languageCode"])] = _confidence(
                            language.get("confidence", 0)
                        )

            if not text:
                continue
            blocks = page.get("blocks", [])
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                paragraphs = block.get("paragraphs", []) if isinstance(block, dict) else []
                if not isinstance(paragraphs, list):
                    continue
                for paragraph in paragraphs:
                    paragraph_words = (
                        paragraph.get("words", []) if isinstance(paragraph, dict) else []
                    )
                    if not isinstance(paragraph_words, list):
                        continue
                    for word in paragraph_words:
                        if not isinstance(word, dict):
                            continue
                        symbols = word.get("symbols", [])
                        if not isinstance(symbols, list):
                            symbols = []
                        word_text = "".join(
                            str(symbol.get("text", ""))
                            for symbol in symbols
                            if isinstance(symbol, dict)
                        )
                        box = word.get("boundingBox", {})
                        vertices = box.get("vertices", []) if isinstance(box, dict) else []
                        if not isinstance(vertices, list):
                            vertices = []
                        coordinates = [
                            [vertex.get("x", 0) or 0, vertex.get("y", 0) or 0]
                            for vertex in vertices
                            if isinstance(vertex, dict)
                        ]
                        words.append(
                            {
                                "text": word_text,
                                "confidence": _confidence(word.get("confidence", 0)),
                                "box": coordinates,
                            }
                        )

    return {
        "text": text,
        "detected_languages": detected_languages,
        "words": words,
    }


def pdf_page_count(pdf_bytes):
    if not pdf_bytes:
        raise ValueError("The PDF is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("The PDF is too large to process. Please upload a smaller document.")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("This PDF is password-protected and cannot be processed.")
            if document.page_count == 0:
                raise ValueError("The PDF does not contain any pages.")
            return document.page_count
    except ValueError:
        raise
    except (fitz.FileDataError, RuntimeError):
        raise ValueError("The uploaded file is not a valid readable PDF.") from None


def _usable_pdf_text(text):
    normalized = " ".join(str(text or "").split())
    return normalized if len(normalized) >= 40 else ""


def _has_substantial_raster_content(page):
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return False

    covered_area = 0.0
    for image in page.get_image_info():
        bbox = image.get("bbox") if isinstance(image, dict) else None
        if not bbox:
            continue
        image_rect = fitz.Rect(bbox)
        left = max(page.rect.x0, image_rect.x0)
        top = max(page.rect.y0, image_rect.y0)
        right = min(page.rect.x1, image_rect.x1)
        bottom = min(page.rect.y1, image_rect.y1)
        if right > left and bottom > top:
            covered_area += (right - left) * (bottom - top)

    return covered_area / page_area >= RASTER_COVERAGE_THRESHOLD


def process_pdf(pdf_bytes, progress_callback=None):
    pdf_page_count(pdf_bytes)

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except (fitz.FileDataError, RuntimeError):
        raise ValueError("The uploaded file is not a valid readable PDF.") from None

    combined_pages = []
    all_words = []
    detected_languages = {}
    page_warnings = []

    try:
        if document.needs_pass:
            raise ValueError("This PDF is password-protected and cannot be processed.")

        page_total = document.page_count
        for page_index in range(page_total):
            page_number = page_index + 1
            if progress_callback:
                progress_callback(page_number, page_total)

            try:
                page = document.load_page(page_index)
                page_text = _usable_pdf_text(page.get_text("text"))
            except Exception as error:
                page_warnings.append(
                    f"Page {page_number} could not be read: {error}"
                )
                continue

            has_raster_content = _has_substantial_raster_content(page)
            if page_text and not has_raster_content:
                combined_pages.append(f"[Page {page_number}]\n{page_text}")
                continue

            try:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                rendered_page = pixmap.tobytes("png")
                page_result = run_ocr(prepare_image(rendered_page))
            except Exception as error:
                page_warnings.append(
                    f"Page {page_number} OCR failed; substantive content may be missing: {error}"
                )
                continue

            ocr_text = str(page_result.get("text", "")).strip()
            if not ocr_text:
                page_warnings.append(
                    f"Page {page_number} OCR produced no usable text; substantive content may be missing."
                )
                continue

            combined_pages.append(f"[Page {page_number}]\n{ocr_text}")
            all_words.extend(page_result.get("words", []))
            for language, confidence in page_result.get("detected_languages", {}).items():
                detected_languages[language] = max(
                    confidence, detected_languages.get(language, 0.0)
                )
    finally:
        document.close()

    return {
        "text": "\n\n".join(combined_pages),
        "detected_languages": detected_languages,
        "words": all_words,
        "page_warnings": page_warnings,
        "page_count": page_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract text from an image with Google Cloud Vision.")
    parser.add_argument("image", help="Path to an image file")
    args = parser.parse_args()

    image_path = Path(args.image)
    result = run_ocr(prepare_image(image_path.read_bytes()))
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"{image_path.stem}.txt").write_text(result["text"], encoding="utf-8")
    (output_dir / f"{image_path.stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result["text"])


if __name__ == "__main__":
    main()
