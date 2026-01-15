from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _match_score(exp: str, found: str) -> float:
    """Return a match score in [0,1].

    - exact => 1.0
    - containment (>=6 chars) => 0.95
    - else SequenceMatcher similarity
    """
    e = _norm(exp)
    f = _norm(found)
    if not e or not f:
        return 0.0
    if e == f:
        return 1.0
    shorter, longer = (e, f) if len(e) <= len(f) else (f, e)
    if len(shorter) >= 6 and shorter in longer:
        return 0.95
    return _sim(e, f)


def build_found_typed(mapping: dict) -> dict:
    """mapping: {"[LABEL_1]": "original value", ...} -> {"LABEL": {values...}}"""
    out = defaultdict(set)
    for tag, val in mapping.items():
        label = tag.strip("[]").rsplit("_", 1)[0]
        out[label].add(val)
    return out


def score_run_typed(mapping: dict, expected_typed: dict, sim_threshold: float = 0.88) -> dict:
    """Compute recall/type accuracy/overall and false positives (same as notebook)."""
    found_typed = build_found_typed(mapping)

    # Flatten expected (type,value)
    expected_pairs = [(label, v) for label, vals in expected_typed.items() for v in vals]
    total_expected = len(expected_pairs)

    # Flatten found (type,value)
    found_pairs = [(label, v) for label, vals in found_typed.items() for v in vals]

    matched_found = set()
    matched_expected = set()
    correct_type_hits = 0

    # Reserved matching: each found can match at most one expected
    for i, (exp_label, exp_val) in enumerate(expected_pairs):
        best_j = None
        best_score = -1.0
        best_type_ok = False

        for j, (found_label, found_val) in enumerate(found_pairs):
            if j in matched_found:
                continue
            score = _match_score(exp_val, found_val)
            if score > best_score:
                best_score = score
                best_j = j
                best_type_ok = found_label == exp_label

        if best_j is not None and best_score >= sim_threshold:
            matched_expected.add(i)
            matched_found.add(best_j)
            if best_type_ok:
                correct_type_hits += 1

    recall = len(matched_expected) / max(1, total_expected)
    type_acc = (correct_type_hits / len(matched_expected)) if matched_expected else 0.0
    overall = recall * type_acc

    # False positives = found pairs not used in any match
    fp_by_type = defaultdict(int)
    fp_total = 0
    for j, (found_label, _found_val) in enumerate(found_pairs):
        if j not in matched_found:
            fp_total += 1
            fp_by_type[found_label] += 1

    return {
        "recall": recall * 100.0,
        "type_accuracy": type_acc * 100.0,
        "overall": overall * 100.0,
        "found_count": len(found_pairs),
        "expected_count": total_expected,
        "false_positives_total": fp_total,
        "false_positives_by_type": dict(fp_by_type),
    }


def print_scores(name: str, s: dict, show_fp_types: bool = True, top_k: int = 12) -> None:
    print(
        f"{name:<10} | "
        f"Recall: {s['recall']:.1f}% | "
        f"TypeAcc: {s['type_accuracy']:.1f}% | "
        f"Overall: {s['overall']:.1f}% | "
        f"FP: {s['false_positives_total']} "
        f"(found {s['found_count']}, expected {s['expected_count']})"
    )
    if show_fp_types and s["false_positives_total"] > 0:
        fp = s["false_positives_by_type"]
        items = sorted(fp.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        fp_str = ", ".join([f"{k}:{v}" for k, v in items])
        print(f"   FP by type: {fp_str}")
