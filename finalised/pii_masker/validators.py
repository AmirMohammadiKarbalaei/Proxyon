from __future__ import annotations

import re

from .normalize import norm_digits, norm_iban, norm_sort_code


def luhn_check(number: str) -> bool:
    digits = re.sub(r"\D+", "", number)
    if len(digits) < 13:
        return False
    total = 0
    alt = False
    for ch in digits[::-1]:
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def iban_mod97(iban: str) -> bool:
    """Basic IBAN mod-97 validation."""
    s = re.sub(r"\s+", "", iban).upper()
    if len(s) < 15:
        return False
    # Move first 4 chars to end
    rearr = s[4:] + s[:4]
    # Convert letters to numbers A=10..Z=35
    converted = ""
    for ch in rearr:
        if ch.isdigit():
            converted += ch
        elif "A" <= ch <= "Z":
            converted += str(ord(ch) - ord("A") + 10)
        else:
            return False
    # mod 97 in chunks
    remainder = 0
    for i in range(0, len(converted), 9):
        chunk = str(remainder) + converted[i : i + 9]
        remainder = int(chunk) % 97
    return remainder == 1


def apply_validators_and_adjust_score(label: str, value: str, base_score: float) -> float:
    """Boost/penalise confidence based on validators."""
    score = float(base_score)

    if label == "CREDIT_CARD_NUMBER":
        if luhn_check(value):
            score = min(1.0, score + 0.08)
        else:
            score = max(0.0, score - 0.15)

    if label == "UK_IBAN":
        if iban_mod97(value):
            score = min(1.0, score + 0.08)
        else:
            score = max(0.0, score - 0.20)

    if label == "UK_SORT_CODE":
        if len(norm_sort_code(value)) == 6:
            score = min(1.0, score + 0.03)
        else:
            score = max(0.0, score - 0.10)

    if label == "UK_ACCOUNT_NUMBER":
        if len(norm_digits(value)) == 8:
            score = min(1.0, score + 0.02)
        else:
            score = max(0.0, score - 0.10)

    return score
