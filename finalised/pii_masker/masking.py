from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .constants import ALLOWED_CANON, GLINER_TO_CANON, PRIORITY
from .normalize import normalise_for_key
from .regex_extractors import extract_regex_spans_v2
from .validators import apply_validators_and_adjust_score


def assign_tags_and_mask(
    text: str,
    spans: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, str], Dict[str, float], List[Dict[str, Any]]]:
    """
    Input spans must already be overlap-resolved and filtered.
    Adds:
      span["tag"]
    Returns:
      masked_text, mapping(tag->original), scores(tag->confidence), spans(with tag)
    """
    counters = defaultdict(int)
    value_to_tag: Dict[Tuple[str, str], str] = {}  # (label, norm_value) -> tag
    mapping: Dict[str, str] = {}
    scores: Dict[str, float] = {}

    # assign tags
    for s in spans:
        label = s["label"]
        original = s["original"]
        key = (label, normalise_for_key(label, original))

        if key in value_to_tag:
            tag = value_to_tag[key]
        else:
            counters[label] += 1
            tag = f"[{label}_{counters[label]}]"
            value_to_tag[key] = tag
            mapping[tag] = original

        # score: keep max across occurrences; adjust with validators
        adj = apply_validators_and_adjust_score(label, original, float(s.get("score", 0.0)))
        scores[tag] = max(scores.get(tag, 0.0), adj)
        s["tag"] = tag
        s["score"] = adj

    # replace from back to front
    masked_text = text
    for s in sorted(spans, key=lambda x: x["start"], reverse=True):
        masked_text = masked_text[: s["start"]] + s["tag"] + masked_text[s["end"] :]

    return masked_text, mapping, scores, spans


def resolve_overlaps_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    spans elements:
      {start,end,label,score,source,original}
    """
    spans_sorted = sorted(
        spans,
        key=lambda s: (
            -PRIORITY.get(s["label"], 0),
            -(s["end"] - s["start"]),
            -float(s.get("score", 0.0)),
            int(s["start"]),
        ),
    )

    kept: List[Dict[str, Any]] = []
    for s in spans_sorted:
        overlaps = False
        for k in kept:
            if not (s["end"] <= k["start"] or s["start"] >= k["end"]):
                overlaps = True
                break
        if not overlaps:
            kept.append(s)

    return sorted(kept, key=lambda s: s["start"])


def mask_with_gliner(
    text: str,
    model_name_or_obj: Any,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Tuple[str, Dict[str, str], Dict[str, float], List[Dict[str, Any]]]:
    """Mask entities in text using GLiNER and deterministic regex backstops.

    Supports:
    - model_name_or_obj: either a GLiNER instance or a model name string

    Expected GLiNER output varies by version; we handle common patterns:
    - list of dicts with keys: start, end, label, score
    """
    try:
        from gliner import GLiNER
    except Exception as e:
        raise RuntimeError("GLiNER not installed. Run: pip install gliner") from e

    if isinstance(model_name_or_obj, str):
        gliner = GLiNER.from_pretrained(model_name_or_obj)
    else:
        gliner = model_name_or_obj

    # If you don't pass labels, use canonical-ish ones (kept identical to notebook)
    if labels is None:
        labels = [
            "person",
            "organization",
            "email_address",
            "phone_number",
            "ip_address",
            "date",
            "address",
            "street_address",
            "location",
            "postcode",
            "uk_iban",
            "sort_code",
            "account_number",
            "credit_card_number",
            "card_expiry",
            "transaction_id",
            "support_ticket_number",
            "session_id",
            "customer_reference",
            "account_id",
        ]

    preds = gliner.predict_entities(text, labels, threshold=threshold)

    spans: List[Dict[str, Any]] = []
    for p in preds:
        raw_label = str(p.get("label", "")).strip()
        canon = GLINER_TO_CANON.get(raw_label.lower(), GLINER_TO_CANON.get(raw_label, None))
        if not canon:
            continue
        if canon not in ALLOWED_CANON:
            continue

        start = int(p["start"])
        end = int(p["end"])
        original = text[start:end]
        spans.append(
            {
                "start": start,
                "end": end,
                "label": canon,
                "score": float(p.get("score", 0.0)),
                "source": "gliner",
                "original": original,
            }
        )

    # Add deterministic regex spans to cover common GLiNER misses
    spans.extend(extract_regex_spans_v2(text))

    spans = resolve_overlaps_spans(spans)
    return assign_tags_and_mask(text, spans)
