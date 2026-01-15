from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple


# --- Regex backstops for common misses (landlines, IPv4, postcodes, address lines) ---
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_UK_POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", re.IGNORECASE)
_UK_LANDLINE_RE = re.compile(r"(?<!\w)0\d{2,4}\s?\d{3,4}\s?\d{3,4}(?!\w)")
_UK_MOBILE_INTL_RE = re.compile(r"(?<!\w)\+44\s?7\d{3}\s?\d{6}(?!\w)")


def _is_valid_ipv4(s: str) -> bool:
    parts = s.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(p) for p in parts]
    except Exception:
        return False
    return all(0 <= n <= 255 for n in nums)


def _expand_to_address_line(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """
    If the span (usually a postcode) sits inside a line that looks like a full address,
    expand to the full line boundaries.
    """
    line_start = text.rfind("\n", 0, start)
    line_start = 0 if line_start == -1 else line_start + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end

    raw_line = text[line_start:line_end]
    line = raw_line.strip()
    if not line:
        return None

    # Heuristics: commas + digits + some street/country-ish token => likely address line
    lc = line.lower()
    if line.count(",") < 2:
        return None
    if not re.search(r"\d", line):
        return None
    if not any(
        tok in lc
        for tok in [
            "street",
            "st",
            "road",
            "rd",
            "avenue",
            "ave",
            "lane",
            "ln",
            "drive",
            "dr",
            "flat",
            "apt",
            "apartment",
            "unit",
            "uk",
            "united kingdom",
            "london",
        ]
    ):
        return None

    # Map back to exact offsets excluding surrounding whitespace
    left_ws = len(raw_line) - len(raw_line.lstrip())
    right_ws = len(raw_line) - len(raw_line.rstrip())
    return line_start + left_ws, line_end - right_ws


def extract_regex_spans_v1(text: str) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []

    # UK phones (landline + +44 mobile)
    for m in _UK_LANDLINE_RE.finditer(text):
        spans.append(
            {
                "start": m.start(),
                "end": m.end(),
                "label": "UK_PHONE_NUMBER",
                "score": 0.99,
                "source": "regex",
                "original": text[m.start() : m.end()],
            }
        )
    for m in _UK_MOBILE_INTL_RE.finditer(text):
        spans.append(
            {
                "start": m.start(),
                "end": m.end(),
                "label": "UK_PHONE_NUMBER",
                "score": 0.99,
                "source": "regex",
                "original": text[m.start() : m.end()],
            }
        )

    # IPv4
    for m in _IPV4_RE.finditer(text):
        s = m.group(0)
        if not _is_valid_ipv4(s):
            continue
        spans.append(
            {
                "start": m.start(),
                "end": m.end(),
                "label": "IP_ADDRESS",
                "score": 0.99,
                "source": "regex",
                "original": s,
            }
        )

    # UK postcode (+ optional full-address line expansion)
    for m in _UK_POSTCODE_RE.finditer(text):
        spans.append(
            {
                "start": m.start(),
                "end": m.end(),
                "label": "UK_POSTCODE",
                "score": 0.99,
                "source": "regex",
                "original": text[m.start() : m.end()],
            }
        )

    return spans


# --- Extra regex backstops: email + dates (incl DOB) ---
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.+-])",
    re.IGNORECASE,
)

_DOB_CONTEXT_RE = re.compile(r"\bDOB\b|\bD\.?O\.?B\.?\b|date\s+of\s+birth|\bborn\b", re.IGNORECASE)

# 12 March 1990 / 17 February 1989 etc.
_DATE_TEXT_RE = re.compile(
    r"\b(?P<day>\d{1,2})\s+"
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)

# 21/12/2025, 21-12-2025, 21.12.2025 (kept strict: requires year)
_DATE_NUMERIC_RE = re.compile(r"\b(?P<day>\d{1,2})[\/\-.](?P<mon>\d{1,2})[\/\-.](?P<year>\d{4})\b")

# 2025/12/21, 2025-12-21, 2025.12.21 (ISO-ish; year first)
_DATE_YMD_RE = re.compile(r"\b(?P<year>\d{4})[\/\-.](?P<mon>\d{1,2})[\/\-.](?P<day>\d{1,2})\b")

_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _is_valid_date_parts(year: int, month: int, day: int) -> bool:
    if year < 1900 or year > 2100:
        return False
    try:
        datetime.date(year, month, day)
    except Exception:
        return False
    return True


def _date_label_for_match(text: str, start: int, end: int) -> str:
    # If "DOB" appears nearby, treat it as DATE_OF_BIRTH; else DATE
    left = text[max(0, start - 40) : start]
    right = text[end : min(len(text), end + 25)]
    ctx = left + " " + right
    return "DATE_OF_BIRTH" if _DOB_CONTEXT_RE.search(ctx) else "DATE"


def extract_regex_spans_v2(text: str) -> List[Dict[str, Any]]:
    # Start with existing regex extraction (phones, IPs, postcodes, address expansion)
    spans = extract_regex_spans_v1(text)

    # Emails
    for m in _EMAIL_RE.finditer(text):
        email = m.group(1)
        spans.append(
            {
                "start": m.start(1),
                "end": m.end(1),
                "label": "EMAIL_ADDRESS",
                "score": 0.99,
                "source": "regex",
                "original": email,
            }
        )

    # Dates (month name)
    for m in _DATE_TEXT_RE.finditer(text):
        day = int(m.group("day"))
        mon_raw = m.group("mon").lower()
        month = _MONTH_MAP.get(mon_raw, _MONTH_MAP.get(mon_raw[:3], 0))
        year = int(m.group("year"))
        if month and _is_valid_date_parts(year, month, day):
            start, end = m.start(), m.end()
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": _date_label_for_match(text, start, end),
                    "score": 0.97,
                    "source": "regex",
                    "original": text[start:end],
                }
            )

    # Dates (numeric with year)
    for m in _DATE_NUMERIC_RE.finditer(text):
        day = int(m.group("day"))
        month = int(m.group("mon"))
        year = int(m.group("year"))
        if _is_valid_date_parts(year, month, day):
            start, end = m.start(), m.end()
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": _date_label_for_match(text, start, end),
                    "score": 0.97,
                    "source": "regex",
                    "original": text[start:end],
                }
            )

    # Dates (numeric year/month/day)
    for m in _DATE_YMD_RE.finditer(text):
        year = int(m.group("year"))
        month = int(m.group("mon"))
        day = int(m.group("day"))
        if _is_valid_date_parts(year, month, day):
            start, end = m.start(), m.end()
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": _date_label_for_match(text, start, end),
                    "score": 0.97,
                    "source": "regex",
                    "original": text[start:end],
                }
            )

    return spans
