import unittest

from tts_normalizer import normalize_for_tts


class TTSNormalizerTests(unittest.TestCase):
    def test_kannada_quantity_is_spoken_naturally(self):
        self.assertIn("ಏಳು ದಿನಗಳ ಒಳಗೆ", normalize_for_tts("7 ದಿನಗಳ ಒಳಗೆ"))

    def test_valid_date_is_spoken_as_month_day_and_year(self):
        result = normalize_for_tts("05-11-2026")
        self.assertIn("ನವೆಂಬರ್ ಐದನೇ ತಾರೀಖು", result)
        self.assertIn("ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು", result)

    def test_invalid_date_is_preserved(self):
        self.assertEqual(normalize_for_tts("31-02-2026"), "31-02-2026")

    def test_times_preserve_existing_grammar(self):
        self.assertEqual(
            normalize_for_tts("ಬೆಳಿಗ್ಗೆ 10:00"), "ಬೆಳಿಗ್ಗೆ ಹತ್ತು ಗಂಟೆಗೆ"
        )
        self.assertEqual(
            normalize_for_tts("5:00 ಗಂಟೆಯ ಒಳಗೆ"), "ಐದು ಗಂಟೆಯ ಒಳಗೆ"
        )

    def test_identifier_uses_digit_by_digit_speech_and_preserves_order(self):
        result = normalize_for_tts("ಆಹಾ/ಪರೀ/2026/1042")
        self.assertIn("ಆಹಾ ಸ್ಲ್ಯಾಶ್ ಪರೀ", result)
        self.assertIn("ಎರಡು ಶೂನ್ಯ ಎರಡು ಆರು", result)
        self.assertIn("ಒಂದು ಶೂನ್ಯ ನಾಲ್ಕು ಎರಡು", result)
        self.assertNotIn("ಆಹಾ/ಪರೀ/2026/1042", result)

    def test_leading_zero_identifier_segment_is_preserved(self):
        result = normalize_for_tts("ಕೇಸ್/005")
        self.assertIn("ಶೂನ್ಯ ಶೂನ್ಯ ಐದು", result)

    def test_known_abbreviations_are_expanded(self):
        result = normalize_for_tts("PDF OCR AI")
        self.assertEqual(result, "ಪಿ ಡಿ ಎಫ್ ಓ ಸಿ ಆರ್ ಎ ಐ")

    def test_mixed_text_and_sample_keep_information(self):
        source = "ಈ ಪತ್ರದ ಸಂಖ್ಯೆ ಆಹಾ/ಪರೀ/2026/1042 ಮತ್ತು ದಿನಾಂಕ 05-11-2026 ಆಗಿದೆ."
        result = normalize_for_tts(source)
        self.assertIn("ಆಹಾ ಸ್ಲ್ಯಾಶ್ ಪರೀ", result)
        self.assertIn("ನವೆಂಬರ್ ಐದನೇ ತಾರೀಖು", result)
        self.assertNotIn("05-11-2026", result)

    def test_ordinary_kannada_and_empty_input_are_preserved(self):
        ordinary = "ಈ ಪತ್ರವನ್ನು ಪರಿಶೀಲಿಸಿ."
        self.assertEqual(normalize_for_tts(ordinary), ordinary)
        self.assertEqual(normalize_for_tts(""), "")
        self.assertIsNone(normalize_for_tts(None))

    def test_normalization_is_idempotent(self):
        source = "ಈ ಪತ್ರದ ಸಂಖ್ಯೆ ಆಹಾ/ಪರೀ/2026/1042 ಮತ್ತು ದಿನಾಂಕ 05-11-2026 ಆಗಿದೆ."
        once = normalize_for_tts(source)
        self.assertEqual(normalize_for_tts(once), once)


if __name__ == "__main__":
    unittest.main()
