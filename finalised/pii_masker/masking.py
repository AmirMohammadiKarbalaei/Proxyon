from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Dict, List, Optional, Tuple

from .constants import ALLOWED_CANON, GLINER_TO_CANON, PRIORITY
from .normalize import normalise_for_key
from .regex_extractors import extract_regex_spans_v2
from .validators import apply_validators_and_adjust_score


_PERSON_ALIAS_STOPWORDS = {
    # Titles / common non-names
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "sir",
    "madam",
    "lord",
    "lady",
    # Months (avoid accidental aliasing like PERSON="May")
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
    # Very common words we never want to alias-mask
    "customer",
    "user",
    "account",
    "support",
}


_PERSON_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# GLiNER tends to under-detect standalone first/last names at higher thresholds.
# We pull more candidates, then apply label-specific thresholds ourselves.
_GLINER_CANDIDATE_THRESHOLD = 0.30
_PERSON_RELAXED_THRESHOLD = 0.45


def _looks_like_name_token(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    # Allow typical name tokens like O'Neil, McDonald, Smith-Jones.
    if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", t):
        return False
    lc = t.lower()
    if lc in _PERSON_ALIAS_STOPWORDS:
        return False
    return True


def _is_titleish(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    # Accept Title Case or ALLCAPS (e.g., in emails/headers).
    return (t[:1].isupper() and t[1:].islower()) or t.isupper()


def _merge_adjacent_person_spans(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge nearby PERSON spans into a single full-name span.

    GLiNER can sometimes emit first/last names as separate spans. Merging them
    improves stability (one tag for the identity) and makes aliasing more reliable.
    """
    if not spans:
        return spans

    # Work on a copy sorted by offsets; only merge PERSON spans.
    sorted_spans = sorted(spans, key=lambda s: (int(s["start"]), -int(s["end"])))
    merged: List[Dict[str, Any]] = []

    i = 0
    while i < len(sorted_spans):
        cur = dict(sorted_spans[i])
        if cur.get("label") != "PERSON":
            merged.append(cur)
            i += 1
            continue

        cur_start = int(cur["start"])
        cur_end = int(cur["end"])
        cur_text = text[cur_start:cur_end]
        if not _looks_like_name_token(cur_text):
            merged.append(cur)
            i += 1
            continue

        j = i + 1
        best_end = cur_end
        best_score = float(cur.get("score", 0.0))
        used = 1

        while j < len(sorted_spans) and used < 4:
            nxt = sorted_spans[j]
            if nxt.get("label") != "PERSON":
                break
            n_start = int(nxt["start"])
            n_end = int(nxt["end"])
            if n_start < best_end:
                # Overlapping PERSON spans are handled elsewhere.
                break

            gap = text[best_end:n_start]
            # Only merge if the gap is tiny and looks like a name separator.
            # Examples: " ", "-", ", ", " " etc.
            if len(gap) > 4:
                break
            if not re.fullmatch(r"[\s,\.-']*", gap):
                break

            n_text = text[n_start:n_end]
            if not _looks_like_name_token(n_text):
                break

            best_end = n_end
            best_score = max(best_score, float(nxt.get("score", 0.0)))
            used += 1
            j += 1

        if best_end != cur_end:
            merged.append(
                {
                    "start": cur_start,
                    "end": best_end,
                    "label": "PERSON",
                    "score": best_score,
                    "source": "gliner_merge" if str(cur.get("source")) == "gliner" else str(cur.get("source", "merge")),
                    "original": text[cur_start:best_end],
                }
            )
            i = j
        else:
            merged.append(cur)
            i += 1

    return merged


def _person_alias_tokens(full_name: str) -> List[str]:
    """Return candidate alias tokens (first/last) derived from a full name."""
    tokens = _PERSON_TOKEN_RE.findall(full_name or "")
    if len(tokens) < 2:
        return []

    first = tokens[0]
    last = tokens[-1]
    out: List[str] = []
    for t in (first, last):
        n = t.strip("'\-").lower()
        if len(n) < 3:
            continue
        if n in _PERSON_ALIAS_STOPWORDS:
            continue
        out.append(t)
    # De-dup but preserve order
    seen = set()
    uniq: List[str] = []
    for t in out:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def _span_overlaps_any(start: int, end: int, spans: List[Dict[str, Any]]) -> bool:
    for s in spans:
        if not (end <= int(s["start"]) or start >= int(s["end"])):
            return True
    return False


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

    # PERSON aliasing: once we see a full name, map first/last tokens -> same tag.
    # This keeps context when later mentions are just "Hannah" or "Mercer".
    person_alias_to_tag: Dict[str, str] = {}  # norm(token) -> existing tag

    # assign tags
    for s in spans:
        label = s["label"]
        original = s["original"]

        # If this looks like a PERSON alias we've already seen, reuse the existing tag.
        if label == "PERSON":
            alias_norm = normalise_for_key("PERSON", original)
            existing = person_alias_to_tag.get(alias_norm)
            if existing is not None:
                tag = existing
                # Keep the max score per tag; still adjust based on validators.
                adj = apply_validators_and_adjust_score(label, original, float(s.get("score", 0.0)))
                scores[tag] = max(scores.get(tag, 0.0), adj)
                s["tag"] = tag
                s["score"] = adj
                continue

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

        # After tagging a full PERSON name, register its first/last token as aliases.
        if label == "PERSON":
            for tok in _person_alias_tokens(original):
                person_alias_to_tag.setdefault(normalise_for_key("PERSON", tok), tag)

    # If we have a PERSON full-name tag, but the model didn't detect later standalone
    # first/last-name occurrences, add deterministic alias spans.
    if person_alias_to_tag:
        # Iterate in decreasing token length to reduce partial-matching ambiguity.
        alias_items = sorted(person_alias_to_tag.items(), key=lambda kv: -len(kv[0]))
        for alias_norm, tag in alias_items:
            # alias_norm is a lowercased token; find it case-insensitively as a whole word.
            # Use \b so "Hannah" in "Hannah's" still masks the name portion.
            if len(alias_norm) < 3 or alias_norm in _PERSON_ALIAS_STOPWORDS:
                continue
            pat = re.compile(rf"\b{re.escape(alias_norm)}\b", re.IGNORECASE)
            for m in pat.finditer(text):
                a_start, a_end = m.start(), m.end()
                if _span_overlaps_any(a_start, a_end, spans):
                    continue
                spans.append(
                    {
                        "start": a_start,
                        "end": a_end,
                        "label": "PERSON",
                        "score": float(scores.get(tag, 0.0)),
                        "source": "alias",
                        "original": text[a_start:a_end],
                        "tag": tag,
                    }
                )

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
            "name",
            "first_name",
            "last_name",
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

    # Ask for more candidates, then filter per label.
    candidate_threshold = min(float(threshold), _GLINER_CANDIDATE_THRESHOLD)
    preds = gliner.predict_entities(text, labels, threshold=candidate_threshold)

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
        score = float(p.get("score", 0.0))

        # Label-specific thresholding: keep global threshold semantics for most
        # labels, but relax for PERSON to improve single-token name recall.
        effective_threshold = float(threshold)
        if canon == "PERSON":
            effective_threshold = min(effective_threshold, _PERSON_RELAXED_THRESHOLD)
        if score < effective_threshold:
            continue
        original = text[start:end]

        # Extra guardrails for single-token PERSON (reduces false positives).
        if canon == "PERSON":
            tokens = _PERSON_TOKEN_RE.findall(original or "")
            if len(tokens) == 1:
                if not _looks_like_name_token(tokens[0]):
                    continue
                if not _is_titleish(tokens[0]):
                    continue
        spans.append(
            {
                "start": start,
                "end": end,
                "label": canon,
                "score": score,
                "source": "gliner",
                "original": original,
            }
        )

    # Add deterministic regex spans to cover common GLiNER misses
    spans.extend(extract_regex_spans_v2(text))

    # Merge adjacent PERSON spans (e.g., first + last name) before overlap resolution.
    spans = _merge_adjacent_person_spans(text, spans)

    spans = resolve_overlaps_spans(spans)
    return assign_tags_and_mask(text, spans)
