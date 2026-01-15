from __future__ import annotations

import re


def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def norm_general(s: str) -> str:
    s = (s or "").lower().strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def norm_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def norm_sort_code(s: str) -> str:
    # 20-45-67, 204567 -> 204567
    return re.sub(r"[^0-9]+", "", (s or ""))


def norm_iban(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").upper().strip())


def norm_phone(s: str) -> str:
    # Keep digits and plus, remove spaces/punct
    s = (s or "").strip()
    s = re.sub(r"[^\d+]+", "", s)
    return s


def normalise_for_key(label: str, value: str) -> str:
    v = (value or "").strip()
    if label == "UK_SORT_CODE":
        return norm_sort_code(v)
    if label == "UK_ACCOUNT_NUMBER":
        return norm_digits(v)
    if label == "UK_IBAN":
        return norm_iban(v)
    if label == "CREDIT_CARD_NUMBER":
        return norm_digits(v)
    if label == "CARD_EXPIRY":
        return norm_general(v)  # "08/27" -> "0827"
    if label == "UK_PHONE_NUMBER":
        return norm_phone(v)
    if label in {"EMAIL_ADDRESS", "IP_ADDRESS"}:
        return norm_general(v)
    return norm_spaces(v).lower()
