import unittest
from io import BytesIO
from unittest.mock import patch

import pymupdf as fitz
from PIL import Image

from ocr import process_pdf


def make_pdf(*page_texts, image_pages=None):
    image_pages = image_pages or {}
    document = fitz.open()
    for page_index, page_text in enumerate(page_texts):
        page = document.new_page()
        if page_text:
            page.insert_text((72, 72), page_text)
        if page_index in image_pages:
            page.insert_image(page.rect, stream=image_pages[page_index])
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class PdfProcessingTests(unittest.TestCase):
    def test_text_pages_are_extracted_in_order_without_ocr(self):
        first = "First page content. " * 4
        second = "Second page content. " * 4

        with patch("ocr.run_ocr") as run_ocr:
            result = process_pdf(make_pdf(first, second))

        self.assertEqual(
            result["text"], f"[Page 1]\n{first.strip()}\n\n[Page 2]\n{second.strip()}"
        )
        run_ocr.assert_not_called()
        self.assertEqual(result["page_warnings"], [])

    def test_scanned_page_uses_existing_image_ocr_pipeline(self):
        ocr_result = {
            "text": "Scanned Kannada page text",
            "words": [{"text": "Scanned", "confidence": 0.9}],
            "detected_languages": {"kn": 0.8},
        }

        with (
            patch("ocr.prepare_image", return_value=b"prepared") as prepare_image,
            patch("ocr.run_ocr", return_value=ocr_result) as run_ocr,
        ):
            result = process_pdf(make_pdf(""))

        self.assertEqual(result["text"], "[Page 1]\nScanned Kannada page text")
        prepare_image.assert_called_once()
        run_ocr.assert_called_once_with(b"prepared")

    def test_large_scanned_image_overrides_selectable_header_text(self):
        header = "eOffice administrative header and footer text. " * 4
        image_buffer = BytesIO()
        Image.new("RGB", (1200, 1600), "white").save(image_buffer, format="PNG")
        ocr_result = {
            "text": "Substantive scanned legal document body",
            "words": [],
            "detected_languages": {"kn": 0.9},
        }

        with patch("ocr.run_ocr", return_value=ocr_result) as run_ocr:
            result = process_pdf(
                make_pdf(header, image_pages={0: image_buffer.getvalue()})
            )

        self.assertIn("Substantive scanned legal document body", result["text"])
        self.assertNotIn("eOffice administrative header", result["text"])
        run_ocr.assert_called_once()

    def test_mixed_pdf_keeps_page_order_and_reports_empty_page(self):
        embedded = "Embedded page content. " * 4
        with patch(
            "ocr.run_ocr",
            return_value={"text": "", "words": [], "detected_languages": {}},
        ):
            result = process_pdf(make_pdf(embedded, ""))

        self.assertEqual(result["text"], f"[Page 1]\n{embedded.strip()}")
        self.assertEqual(
            result["page_warnings"],
            [
                "Page 2 OCR produced no usable text; substantive content may be missing."
            ],
        )

    def test_ocr_failure_does_not_fall_back_to_embedded_header_text(self):
        header = "Selectable administrative header and footer text. " * 4
        image_buffer = BytesIO()
        Image.new("RGB", (1200, 1600), "white").save(image_buffer, format="PNG")

        with patch("ocr.run_ocr", side_effect=RuntimeError("Vision unavailable")):
            result = process_pdf(
                make_pdf(header, image_pages={0: image_buffer.getvalue()})
            )

        self.assertEqual(result["text"], "")
        self.assertIn("substantive content may be missing", result["page_warnings"][0])

    def test_invalid_and_empty_pdfs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            process_pdf(b"")
        with self.assertRaisesRegex(ValueError, "valid readable PDF"):
            process_pdf(b"not a pdf")


if __name__ == "__main__":
    unittest.main()
