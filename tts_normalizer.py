"""Deterministic speech-oriented normalization for Kannada TTS."""

from datetime import date
import re


_ONES = (
    "ಶೂನ್ಯ",
    "ಒಂದು",
    "ಎರಡು",
    "ಮೂರು",
    "ನಾಲ್ಕು",
    "ಐದು",
    "ಆರು",
    "ಏಳು",
    "ಎಂಟು",
    "ಒಂಬತ್ತು",
    "ಹತ್ತು",
    "ಹನ್ನೊಂದು",
    "ಹನ್ನೆರಡು",
    "ಹದಿಮೂರು",
    "ಹದಿನಾಲ್ಕು",
    "ಹದಿನೈದು",
    "ಹದಿನಾರು",
    "ಹದಿನೇಳು",
    "ಹದಿನೆಂಟು",
    "ಹತ್ತೊಂಬತ್ತು",
)
_TENS = {
    20: "ಇಪ್ಪತ್ತು",
    30: "ಮೂವತ್ತು",
    40: "ನಲವತ್ತು",
    50: "ಐವತ್ತು",
    60: "ಅರವತ್ತು",
    70: "ಎಪ್ಪತ್ತು",
    80: "ಎಂಬತ್ತು",
    90: "ತೊಂಬತ್ತು",
}
_TENS_STEMS = {
    20: "ಇಪ್ಪತ್ತ",
    30: "ಮೂವತ್ತ",
    40: "ನಲವತ್ತ",
    50: "ಐವತ್ತ",
    60: "ಅರವತ್ತ",
    70: "ಎಪ್ಪತ್ತ",
    80: "ಎಂಬತ್ತ",
    90: "ತೊಂಬತ್ತ",
}
_ORDINALS = {
    1: "ಮೊದಲನೇ",
    2: "ಎರಡನೇ",
    3: "ಮೂರನೇ",
    4: "ನಾಲ್ಕನೇ",
    5: "ಐದನೇ",
    6: "ಆರನೇ",
    7: "ಏಳನೇ",
    8: "ಎಂಟನೇ",
    9: "ಒಂಬತ್ತನೇ",
    10: "ಹತ್ತನೇ",
    11: "ಹನ್ನೊಂದನೇ",
    12: "ಹನ್ನೆರಡನೇ",
    13: "ಹದಿಮೂರನೇ",
    14: "ಹದಿನಾಲ್ಕನೇ",
    15: "ಹದಿನೈದನೇ",
    16: "ಹದಿನಾರನೇ",
    17: "ಹದಿನೇಳನೇ",
    18: "ಹದಿನೆಂಟನೇ",
    19: "ಹತ್ತೊಂಬತ್ತನೇ",
    20: "ಇಪ್ಪತ್ತನೇ",
    21: "ಇಪ್ಪತ್ತೊಂದನೇ",
    22: "ಇಪ್ಪತ್ತೆರಡನೇ",
    23: "ಇಪ್ಪತ್ತಮೂರನೇ",
    24: "ಇಪ್ಪತ್ತನಾಲ್ಕನೇ",
    25: "ಇಪ್ಪತ್ತೈದನೇ",
    26: "ಇಪ್ಪತ್ತಾರನೇ",
    27: "ಇಪ್ಪತ್ತೇಳನೇ",
    28: "ಇಪ್ಪತ್ತೆಂಟನೇ",
    29: "ಇಪ್ಪತ್ತೊಂಬತ್ತನೇ",
    30: "ಮೂವತ್ತನೇ",
    31: "ಮೂವತ್ತೊಂದನೇ",
}
_MONTHS = {
    1: "ಜನವರಿ",
    2: "ಫೆಬ್ರವರಿ",
    3: "ಮಾರ್ಚ್",
    4: "ಏಪ್ರಿಲ್",
    5: "ಮೇ",
    6: "ಜೂನ್",
    7: "ಜುಲೈ",
    8: "ಆಗಸ್ಟ್",
    9: "ಸೆಪ್ಟೆಂಬರ್",
    10: "ಅಕ್ಟೋಬರ್",
    11: "ನವೆಂಬರ್",
    12: "ಡಿಸೆಂಬರ್",
}
_DIGITS = ("ಶೂನ್ಯ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು")
_ABBREVIATIONS = {
    "PDF": "ಪಿ ಡಿ ಎಫ್",
    "OCR": "ಓ ಸಿ ಆರ್",
    "AI": "ಎ ಐ",
    "TTS": "ಟಿ ಟಿ ಎಸ್",
    "STT": "ಎಸ್ ಟಿ ಟಿ",
    "API": "ಎ ಪಿ ಐ",
    "URL": "ಯು ಆರ್ ಎಲ್",
    "ID": "ಐ ಡಿ",
}

_IDENTIFIER_RE = re.compile(
    r"(?<!\S)((?=[^\s,;.!?]*\d)[^\s,;.!?]+(?:/[^\s,;.!?]+)+)",
    re.UNICODE,
)
_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4})(?!\d)")
_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):([0-5]\d)(?:\s*(ಗಂಟೆಯಿಂದ|ಗಂಟೆಗೆ|ಗಂಟೆಯ))?(?!\d)")
_ABBREVIATION_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(_ABBREVIATIONS) + r")(?![A-Za-z])",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w/:-])(\d+)(?![\w/:-])", re.UNICODE)
_NUMBER_SUFFIX_RE = re.compile(r"(?<![\w/:-])(\d+)(?=(?:ರ|ನೇ|ನೆಯ))", re.UNICODE)


def _number_to_kannada(value: int) -> str:
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens = value // 10 * 10
        remainder = value % 10
        if not remainder:
            return _TENS[tens]
        suffix = {
            6: "ಾರು",
            8: "ೆಂಟು",
            9: "ೊಂಬತ್ತು",
        }.get(remainder, _ONES[remainder])
        return _TENS_STEMS[tens] + suffix
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        prefix = "ನೂರು" if hundreds == 1 else f"{_ONES[hundreds]} ನೂರು"
        return prefix if not remainder else f"{prefix} {_number_to_kannada(remainder)}"
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        prefix = "ಸಾವಿರ" if thousands == 1 else f"{_number_to_kannada(thousands)} ಸಾವಿರ"
        return prefix if not remainder else f"{prefix}ದ {_number_to_kannada(remainder)}"
    return str(value)


def _digits_to_kannada(value: str) -> str:
    return " ".join(_DIGITS[int(digit)] for digit in value)


def _replace_identifier(match: re.Match) -> str:
    parts = match.group(1).split("/")
    spoken_parts = []
    for part in parts:
        spoken_parts.append(_digits_to_kannada(part) if part.isdigit() else part)
    return " ಸ್ಲ್ಯಾಶ್ ".join(spoken_parts)


def _replace_date(match: re.Match) -> str:
    first, second, third = (int(group) for group in match.groups())
    day, month, year = first, second, third
    try:
        date(year, month, day)
    except ValueError:
        return match.group(0)
    return f"{_MONTHS[month]} {_ORDINALS[day]} ತಾರೀಖು, {_number_to_kannada(year)}"


def _replace_iso_date(match: re.Match) -> str:
    year, month, day = (int(group) for group in match.groups())
    try:
        date(year, month, day)
    except ValueError:
        return match.group(0)
    return f"{_MONTHS[month]} {_ORDINALS[day]} ತಾರೀಖು, {_number_to_kannada(year)}"


def _replace_time(match: re.Match) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23:
        return match.group(0)
    hour_text = _number_to_kannada(hour)
    suffix = match.group(3)
    if suffix:
        if minute == 0:
            return f"{hour_text} {suffix}"
        return f"{hour_text} ಗಂಟೆ {_number_to_kannada(minute)} ನಿಮಿಷ {suffix.replace('ಗಂಟೆ', '')}".strip()
    if minute == 0:
        return f"{hour_text} ಗಂಟೆಗೆ"
    return f"{hour_text} ಗಂಟೆ {_number_to_kannada(minute)} ನಿಮಿಷಕ್ಕೆ"


def normalize_for_tts(text: str) -> str:
    """Normalize only unambiguous speech patterns, leaving source text unchanged."""
    if not text or not text.strip():
        return text

    normalized = str(text)
    normalized = _IDENTIFIER_RE.sub(_replace_identifier, normalized)
    normalized = _ISO_DATE_RE.sub(_replace_iso_date, normalized)
    normalized = _DATE_RE.sub(_replace_date, normalized)
    normalized = _TIME_RE.sub(_replace_time, normalized)
    normalized = _ABBREVIATION_RE.sub(
        lambda match: _ABBREVIATIONS[match.group(1).upper()], normalized
    )
    normalized = _NUMBER_RE.sub(lambda match: _number_to_kannada(int(match.group(1))), normalized)
    normalized = _NUMBER_SUFFIX_RE.sub(
        lambda match: _number_to_kannada(int(match.group(1))), normalized
    )
    return normalized
