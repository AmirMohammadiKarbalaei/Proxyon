from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .masking import mask_with_gliner
from .scoring import print_scores, score_run_typed


_MODEL_CACHE: Dict[str, Any] = {}


def _get_gliner_model(model_name: str) -> Any:
    """Load and cache GLiNER model by name (so repeated manual tests are fast)."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from gliner import GLiNER
    except RuntimeError as e:
        msg = str(e)
        if "TORCH_LIBRARY" in msg and "prims" in msg:
            raise RuntimeError(
                "PyTorch is in a corrupted import state (duplicate prims registration). "
                "In VS Code: restart the notebook kernel and re-run the import cell. "
                "Also avoid installing/upgrading torch while the kernel is running."
            ) from e
        raise
    except Exception as e:
        raise RuntimeError(
            "GLiNER is not available in this environment. Install it with `pip install gliner` "
            "(recommended on Python 3.10–3.12)."
        ) from e

    model = GLiNER.from_pretrained(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def run_test(
    text: str,
    model_name: str = "urchade/gliner_multi_pii-v1",
    threshold: float = 0.5,
) -> Tuple[str, Dict[str, str], Dict[str, float], List[Dict[str, Any]]]:
    """Run the full masking pipeline on a single string.

    This is the same masking used by `run_tests_from_json`, but takes `text` directly.
    Returns: masked_text, mapping, scores, spans.
    """
    model = _get_gliner_model(model_name)
    return mask_with_gliner(text=text, model_name_or_obj=model, threshold=threshold)


def load_tests_bundle(path: str | Path = "tests.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    if not isinstance(bundle, dict) or "tests" not in bundle:
        raise ValueError(f"Unexpected format in {p}")
    return bundle


def expected_typed_to_sets(expected_typed: Dict[str, List[str]]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for label, values in (expected_typed or {}).items():
        out[label] = set(values or [])
    return out


def run_tests_from_json(
    model_name: str = "urchade/gliner_multi_pii-v1",
    path: str | Path = "tests.json",
    threshold: float = 0.5,
    show_masked_text: bool = False,
    limit: int | None = None,
) -> List[Tuple[str, dict, str]]:
    bundle = load_tests_bundle(path)
    tests = bundle.get("tests", [])
    if not isinstance(tests, list):
        raise ValueError("bundle['tests'] must be a list")

    try:
        from gliner import GLiNER
    except RuntimeError as e:
        msg = str(e)
        if "TORCH_LIBRARY" in msg and "prims" in msg:
            raise RuntimeError(
                "PyTorch is in a corrupted import state (duplicate prims registration). "
                "In VS Code: restart the notebook kernel, then run the test cells again. "
                "Also avoid installing/upgrading torch while the kernel is running."
            ) from e
        raise

    model = GLiNER.from_pretrained(model_name)

    if limit is not None:
        tests = tests[: int(limit)]

    results: List[Tuple[str, dict, str]] = []
    print(f"Loaded {len(tests)} tests from {Path(path).resolve()}")
    print(f"Model: {model_name} | threshold={threshold}")
    print("-")

    for t in tests:
        test_id = str(t.get("id", "<no-id>"))
        text = str(t.get("text", ""))
        expected_typed_raw = t.get("expected_typed", {}) or {}
        expected_typed = expected_typed_to_sets(expected_typed_raw)

        masked_text, mapping, _scores, _spans = mask_with_gliner(
            text=text,
            model_name_or_obj=model,
            threshold=threshold,
        )
        s = score_run_typed(mapping=mapping, expected_typed=expected_typed)
        print_scores(test_id, s)
        if show_masked_text:
            print("Masked text:")
            print(masked_text)
            print("-")
        results.append((test_id, s, masked_text))

    # Simple aggregate (macro average)
    if results:
        avg = {
            "recall": sum(r[1]["recall"] for r in results) / len(results),
            "type_accuracy": sum(r[1]["type_accuracy"] for r in results) / len(results),
            "overall": sum(r[1]["overall"] for r in results) / len(results),
            "false_positives_total": sum(r[1]["false_positives_total"] for r in results),
            "found_count": sum(r[1]["found_count"] for r in results),
            "expected_count": sum(r[1]["expected_count"] for r in results),
            "false_positives_by_type": {},
        }
        print("=")
        print_scores("AVG", avg, show_fp_types=False)

    return results
