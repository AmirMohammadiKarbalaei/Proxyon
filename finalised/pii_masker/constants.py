from __future__ import annotations

# Canonical labels that we will emit.
CANON_LABELS = {
    # Identity
    "PERSON",
    "ORG",

    # Contact
    "EMAIL_ADDRESS",
    "UK_PHONE_NUMBER",
    "IP_ADDRESS",

    # Time
    "DATE",
    "DATE_OF_BIRTH",

    # Geography (privacy-safe)
    "LOCATION",
    "UK_POSTCODE",
    "UK_ADDRESS",

    # Banking
    "UK_SORT_CODE",
    "UK_ACCOUNT_NUMBER",
    "UK_IBAN",

    # Cards
    "CREDIT_CARD_NUMBER",
    "CARD_EXPIRY",

    # IDs
    "TRANSACTION_ID",
    "CUSTOMER_REFERENCE",
    "SESSION_ID",
    "SUPPORT_TICKET_NUMBER",
    "ACCOUNT_ID",
    "INTERNAL_ID",
}

ALLOWED_CANON = set(CANON_LABELS)

# Map GLiNER labels (lowercased) to canonical labels.
# Kept identical to the notebook mapping.
GLINER_TO_CANON = {
    # ---- People & Orgs ----
    "person": "PERSON",
    "name": "PERSON",
    "first_name": "PERSON",
    "last_name": "PERSON",
    "organization": "ORG",
    "organisation": "ORG",
    "company": "ORG",
    "org": "ORG",

    # ---- Contact ----
    "email": "EMAIL_ADDRESS",
    "email_address": "EMAIL_ADDRESS",
    "phone": "UK_PHONE_NUMBER",
    "phone_number": "UK_PHONE_NUMBER",
    "mobile": "UK_PHONE_NUMBER",

    # ---- Network ----
    "ip": "IP_ADDRESS",
    "ip_address": "IP_ADDRESS",

    # ---- Dates ----
    "date": "DATE",
    "date_time": "DATE",
    "datetime": "DATE",

    # ---- Geography (privacy-safe) ----
    # Treat full address as its own PII type when available, otherwise fall back to
    # postcode/street/city/etc as LOCATION/UK_POSTCODE.
    "address": "UK_ADDRESS",
    "street_address": "UK_ADDRESS",
    "full_address": "UK_ADDRESS",

    "location": "LOCATION",
    "city": "LOCATION",
    "town": "LOCATION",
    "state": "LOCATION",
    "province": "LOCATION",
    "region": "LOCATION",
    "country": "LOCATION",
    "place": "LOCATION",

    # ---- UK specific ----
    "postcode": "UK_POSTCODE",
    "uk_postcode": "UK_POSTCODE",

    # ---- Banking ----
    "uk_iban": "UK_IBAN",
    "iban": "UK_IBAN",
    "sort_code": "UK_SORT_CODE",
    "uk_sort_code": "UK_SORT_CODE",
    "account_number": "UK_ACCOUNT_NUMBER",
    "uk_account_number": "UK_ACCOUNT_NUMBER",

    # ---- Cards ----
    "credit_card_number": "CREDIT_CARD_NUMBER",
    "card_number": "CREDIT_CARD_NUMBER",
    "card_expiry": "CARD_EXPIRY",
    "expiry": "CARD_EXPIRY",
    "expiration_date": "CARD_EXPIRY",

    # ---- IDs ----
    "transaction_id": "TRANSACTION_ID",
    "support_ticket_number": "SUPPORT_TICKET_NUMBER",
    "session_id": "SESSION_ID",
    "customer_reference": "CUSTOMER_REFERENCE",
    "account_id": "ACCOUNT_ID",
    "internal_id": "INTERNAL_ID",
}

# Overlap resolution priority (higher wins). Kept identical to the notebook.
PRIORITY = {
    "UK_IBAN": 120,
    "CREDIT_CARD_NUMBER": 115,
    "UK_SORT_CODE": 110,
    "UK_ACCOUNT_NUMBER": 108,
    "CARD_EXPIRY": 105,

    "EMAIL_ADDRESS": 95,
    "IP_ADDRESS": 95,
    "UK_PHONE_NUMBER": 92,

    "UK_ADDRESS": 88,
    "UK_POSTCODE": 85,

    "TRANSACTION_ID": 75,
    "SUPPORT_TICKET_NUMBER": 74,
    "SESSION_ID": 73,
    "CUSTOMER_REFERENCE": 72,
    "ACCOUNT_ID": 71,
    "INTERNAL_ID": 70,

    "DATE_OF_BIRTH": 55,
    "DATE": 50,
    "PERSON": 40,
    "ORG": 35,
}
