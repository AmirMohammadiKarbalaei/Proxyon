# How this PII masker works (deep dive)

This workspace contains two closely-related implementations of the same idea:

1. A **notebook prototype** in `test.ipynb`.
2. A **packaged implementation** in `finalised/pii_masker/`.

They implement a **PII masking pipeline** that:

- Runs a GLiNER model to detect entities in free text.
- Adds deterministic **regex backstops** for common PII patterns.
- Resolves overlapping detections using a **priority-based chooser**.
- Converts final spans into stable placeholder tags like `[EMAIL_ADDRESS_1]`.
- Returns:
  - `masked_text`: the text with placeholders inserted,
  - `mapping`: placeholder → original value,
  - `scores`: placeholder → confidence,
  - `spans`: the chosen spans including offsets, label, source, etc.

It also includes a **scoring harness** that compares detected items to expected items in `tests.json` and reports recall, type accuracy, and false positives.

---

## Quick map of the important code

Packaged implementation is in `finalised/pii_masker/`:

- `constants.py`
  - Defines canonical labels the system can emit.
  - Defines mapping from GLiNER labels to canonical labels.
  - Defines overlap priority values per canonical label.

- `regex_extractors.py`
  - Produces spans via regex for PII that GLiNER commonly misses.

- `masking.py`
  - Main pipeline: GLiNER predictions + regex spans → overlap resolution → tagging/masking.

- `normalize.py`
  - Defines normalization rules used to **deduplicate** repeated values under one tag.

- `validators.py`
  - Optional “sanity checks” (Luhn, IBAN mod-97, length checks) used to adjust confidence.

- `scoring.py`
  - Compares found values to expected values with a lenient matcher.

- `tests_runner.py`
  - Loads `tests.json`, runs the model, prints per-test metrics and averages.

Dependencies:

- `finalised/requirements.txt` currently only lists `gliner`.
  - GLiNER pulls heavy transitive dependencies (notably `torch`).

---

## The core idea: spans + deterministic replacement

Everything revolves around a **span**.

A span is a dictionary with (at minimum) these fields:

- `start` (int): character offset into the original `text` where the entity starts.
- `end` (int): character offset where the entity ends (Python slice end, exclusive).
- `label` (str): canonical PII type (e.g. `EMAIL_ADDRESS`).
- `score` (float): confidence score in `[0, 1]`.
- `source` (str): where it came from, typically `"gliner"` or `"regex"`.
- `original` (str): `text[start:end]` captured at extraction time.

After overlap resolution, an additional field is added:

- `tag` (str): placeholder like `[EMAIL_ADDRESS_1]`.

The pipeline is deliberately **offset-based**. It never does “find/replace by string” in the main masking step because:

- the same substring may appear multiple times,
- substrings may be ambiguous,
- substring replacement can break when punctuation/whitespace differs,
- span offsets preserve exactness.

---

## Canonical labels and label mapping

### Canonical labels

The system only emits a fixed set of PII types defined in `constants.py` as `CANON_LABELS`.

Examples include:

- Identity: `PERSON`, `ORG`
- Contact: `EMAIL_ADDRESS`, `UK_PHONE_NUMBER`, `IP_ADDRESS`
- Time: `DATE`, `DATE_OF_BIRTH`
- Geography: `LOCATION`, `UK_POSTCODE`, `UK_ADDRESS`
- Banking: `UK_SORT_CODE`, `UK_ACCOUNT_NUMBER`, `UK_IBAN`
- Cards: `CREDIT_CARD_NUMBER`, `CARD_EXPIRY`
- IDs: `TRANSACTION_ID`, `CUSTOMER_REFERENCE`, `SESSION_ID`, `SUPPORT_TICKET_NUMBER`, `ACCOUNT_ID`, `INTERNAL_ID`

This “closed set” is important because downstream logic (priority, normalization, validators, and scoring) are keyed by canonical label names.

### Mapping GLiNER labels to canonical labels

GLiNER may return labels like `person`, `phone_number`, `postcode`, etc.

The mapping `GLINER_TO_CANON` converts those into canonical labels. For example:

- `person` → `PERSON`
- `phone_number` → `UK_PHONE_NUMBER`
- `address` → `UK_ADDRESS`
- `sort_code` → `UK_SORT_CODE`

In `masking.mask_with_gliner`, each GLiNER prediction label is:

1. converted to string,
2. stripped,
3. looked up case-insensitively:
   - `GLINER_TO_CANON.get(raw_label.lower(), GLINER_TO_CANON.get(raw_label, None))`
4. rejected if not in `ALLOWED_CANON`.

This means:

- Unknown labels are ignored.
- Even if GLiNER predicts something, it won’t be emitted unless it maps to the canonical allowlist.

---

## Step-by-step masking pipeline (the exact data flow)

This section describes exactly what happens when you call:

- `pii_masker.masking.mask_with_gliner(text, model_name_or_obj, threshold=...)`

### Step 0: Load model

`mask_with_gliner` accepts either:

- `model_name_or_obj` as a string (e.g. `"urchade/gliner_multi_pii-v1"`), or
- an already-loaded model instance.

If a string is provided, it calls:

- `GLiNER.from_pretrained(model_name_or_obj)`

This downloads / loads the model.

### Step 1: Choose candidate labels (the prompt labels)

GLiNER is prompted with a list of labels:

```text
person, organization,
email_address, phone_number, ip_address,
date,
address, street_address,
location,
postcode,
uk_iban, sort_code, account_number,
credit_card_number, card_expiry,
transaction_id, support_ticket_number,
session_id, customer_reference, account_id
```

These are *not* the canonical labels; they are the labels GLiNER will attempt to predict.

If you pass `labels=...` explicitly, that list is used instead.

### Step 2: Run GLiNER inference

`preds = gliner.predict_entities(text, labels, threshold=threshold)`

`threshold` is passed straight to GLiNER. Higher threshold typically means fewer predictions.

`preds` is expected to be a list of dicts that contain:

- `start`, `end` (offsets)
- `label`
- `score`

(If GLiNER changes output format, this code would be the first place to update.)

### Step 3: Convert GLiNER predictions into spans

For each prediction dict `p`, the code:

1. Maps `p["label"]` to a canonical label using `GLINER_TO_CANON`.
2. Ensures label is in the allowlist.
3. Extracts `start = int(p["start"])`, `end = int(p["end"])`.
4. Computes `original = text[start:end]`.
5. Appends a span dict:

```python
{
  "start": start,
  "end": end,
  "label": canon,
  "score": float(p.get("score", 0.0)),
  "source": "gliner",
  "original": original,
}
```

At this point, spans can overlap heavily because:

- GLiNER may predict nested entities,
- e.g. a postcode inside an address,
- or account numbers inside IBANs,
- or a person’s name inside an email address.

### Step 4: Add regex backstop spans

After GLiNER spans are collected, regex spans are appended:

- `spans.extend(extract_regex_spans_v2(text))`

`extract_regex_spans_v2` includes all patterns from `v1` plus:

- emails
- dates (month-name dates, numeric D/M/Y, numeric Y/M/D)
- DOB-sensitive relabeling (DATE vs DATE_OF_BIRTH)

Regex spans are intentionally assigned very high confidence (0.97–0.99) because they are deterministic pattern matches.

Important detail: **regex spans are not overlap-resolved inside the extractor** in the packaged version; they are added raw and overlap-resolved later by the global resolver.

### Step 5: Resolve overlaps globally

`resolve_overlaps_spans(spans)` selects a subset of non-overlapping spans.

#### 5.1 Sorting rule

The resolver sorts spans by a tuple:

1. `-PRIORITY[label]` (higher priority first)
2. `-(length)` where length = `end - start` (longer spans first)
3. `-score` (higher confidence first)
4. `start` (earlier spans first)

This is a deterministic ordering: given the same input spans, it produces the same order.

#### 5.2 Selection rule

It iterates through spans in that sorted order and greedily keeps a span if it does not overlap any already kept span.

Overlap is defined as:

- Two spans overlap if their ranges intersect in any character.
- Non-overlapping means:
  - `s.end <= k.start` (s ends before k starts) OR
  - `s.start >= k.end` (s starts after k ends)

If neither is true, they overlap.

#### 5.3 Why priority exists

Priority encodes domain decisions such as:

- If an IBAN overlaps an account number, keep the IBAN.
- If a postcode appears inside an address line, keep the address (if it was detected).

Because `PRIORITY` is the first sort key, it dominates the selection.

#### 5.4 Output ordering

After choosing non-overlapping spans, it returns them sorted by increasing `start` offset.

That final ordering matters for later steps.

### Step 6: Assign tags (deduping repeated values)

`assign_tags_and_mask(text, spans)` assigns stable placeholder tags.

It maintains:

- `counters[label]` → how many tags we have created for each label.
- `value_to_tag[(label, norm_value)]` → reuses a tag if the same logical value appears again.

The normalization function is label-aware:

- `UK_SORT_CODE`: remove all non-digits
- `UK_ACCOUNT_NUMBER`: digits only
- `UK_IBAN`: remove spaces, uppercase
- `CREDIT_CARD_NUMBER`: digits only
- `CARD_EXPIRY`: remove non-alphanumerics, lowercase (so `08/27` and `08-27` match)
- `UK_PHONE_NUMBER`: keep digits and `+`
- `EMAIL_ADDRESS`, `IP_ADDRESS`: normalized alphanumeric (this is intentionally aggressive)
- default: collapse whitespace and lowercase

This is not “validation”; it’s used only for **deduplication** so that repeated mentions get the same tag.

#### Tag format

Tags look like:

- `[LABEL_N]`

Where:

- `LABEL` is the canonical label.
- `N` increments per label.

So the first email is `[EMAIL_ADDRESS_1]`, the second distinct email is `[EMAIL_ADDRESS_2]`, etc.

If the same email appears again, it reuses the tag from `value_to_tag` and does not create a new one.

### Step 7: Adjust confidence scores with validators

Each span’s score is adjusted via:

- `apply_validators_and_adjust_score(label, original, base_score)`

This does *not* decide whether to keep or drop the span; it only modifies the confidence.

Current validator rules:

- `CREDIT_CARD_NUMBER`
  - Luhn check passes: +0.08 (capped at 1.0)
  - fails: -0.15 (floored at 0.0)

- `UK_IBAN`
  - mod-97 passes: +0.08
  - fails: -0.20

- `UK_SORT_CODE`
  - exactly 6 digits after normalization: +0.03
  - else: -0.10

- `UK_ACCOUNT_NUMBER`
  - exactly 8 digits after normalization: +0.02
  - else: -0.10

Scores are stored two ways:

- Per span: `s["score"] = adj`.
- Per tag: `scores[tag] = max(existing, adj)` (keeps the highest confidence among repeated occurrences).

### Step 8: Perform masking replacement safely

Replacement is done from **right to left**:

- spans are sorted by `start` descending
- each replacement does:
  - `masked_text = masked_text[:start] + tag + masked_text[end:]`

Right-to-left replacement avoids a common bug: if you replace earlier spans first, you shift all later offsets and corrupt the replacement boundaries.

Because overlaps have already been resolved, no two spans will compete for the same characters during replacement.

### Step 9: Return outputs

`mask_with_gliner` returns:

1. `masked_text` (str)
2. `mapping` (dict tag → original)
3. `scores` (dict tag → float confidence)
4. `spans` (list of span dicts, now including `tag` and adjusted `score`)

---

## Regex backstops: what they do (and exactly how)

Regex logic exists because model-based detection has typical “blind spots”:

- phone number formatting variability
- postcodes
- IPv4 addresses
- dates
- emails
- addresses that span a whole line

### Phone numbers

Patterns:

- UK landlines: `0` then 2–4 digits, then blocks of 3–4 digits separated by optional spaces.
- International UK mobiles: `+44` then `7xxx xxxxxx`.

All produce label `UK_PHONE_NUMBER`, score `0.99`.

### IPv4

Pattern matches `d.d.d.d` with 1–3 digits per segment.

Then `_is_valid_ipv4` ensures each segment is 0–255.

Produces label `IP_ADDRESS`, score `0.99`.

### UK postcodes and address expansion

Postcode pattern matches many typical UK postcode formats.

Note: in the current packaged implementation, **postcode-based address-line expansion is disabled** (i.e., it emits `UK_POSTCODE` but does not auto-expand to a whole-line `UK_ADDRESS`).

### Emails

Email regex matches common email patterns with basic boundary checks.

Produces label `EMAIL_ADDRESS`, score `0.99`.

### Dates and DOB relabeling

Three date styles are recognized:

- `12 March 1990` (month name)
- `21/12/2025` (day/month/year with separator `/`, `-`, `.`)
- `2025-12-21` (year-month-day, “ISO-ish”)

All of these are validated by constructing a `datetime.date` and bounding year to 1900–2100.

Then `_date_label_for_match` inspects surrounding context (40 chars left, 25 chars right). If it sees:

- `DOB`, `D.O.B`, `date of birth`, or `born`

it labels the span as `DATE_OF_BIRTH`; otherwise `DATE`.

---

## Scoring and tests: how correctness is measured

The tests live in `tests.json` (both in workspace root and `finalised/tests.json`).

Each test is expected to have:

- `id`
- `text`
- `expected_typed` (dict): label → list of expected values

Example shape:

```json
{
  "id": "T001",
  "text": "Contact me at alice@example.com",
  "expected_typed": {
    "EMAIL_ADDRESS": ["alice@example.com"]
  }
}
```

### What the pipeline returns for scoring

Scoring uses `mapping` (tag → original). This is converted to a typed structure:

- `build_found_typed(mapping)` → `{ "LABEL": {values...} }`

It infers label from the placeholder tag by stripping brackets and removing the numeric suffix:

- `[EMAIL_ADDRESS_1]` → `EMAIL_ADDRESS`

### Matching is intentionally lenient

To decide whether an expected value matches a found value, it uses `_match_score(exp, found)`:

1. Normalize both strings:
   - lowercase, trim
   - remove all non `[a-z0-9]`

2. If normalized strings equal → match score 1.0

3. If one normalized string contains the other and the shorter is at least 6 chars → 0.95

4. Else, use `SequenceMatcher` similarity ratio.

A match is accepted if score ≥ `sim_threshold` (default 0.88).

This matcher is forgiving by design because:

- masking spans may include slightly different punctuation/whitespace,
- GLiNER may return a superset substring,
- you may store expected values in a normalized form.

### “Reserved matching” (one-to-one pairing)

Each found value can match at most one expected value.

For each expected value, the scorer finds the best unmatched found value.

This prevents “one found value” from incorrectly satisfying many expected items.

### Metrics

For each test:

- `recall` = (matched_expected / total_expected) × 100
- `type_accuracy` = (correct_type_hits / matched_expected) × 100
- `overall` = recall × type_accuracy / 100 (implemented as `recall_fraction * type_acc_fraction`, then ×100)
- `false_positives_total` = number of found values not matched to any expected value
- `false_positives_by_type` = histogram by label

Important nuance:

- It is possible to have high recall but low type accuracy if values are found but labeled incorrectly.

---

## How to run it (notebook and package)

### Option A: run packaged code

From the `finalised/` folder, install deps and run:

- `pip install -r requirements.txt`

Then in Python:

```python
from pii_masker.tests_runner import run_tests_from_json
run_tests_from_json(path="tests.json", threshold=0.1)
```

Or run a single string:

```python
from pii_masker.tests_runner import run_test
masked_text, mapping, scores, spans = run_test("Email alice@example.com", threshold=0.1)
```

### Option B: run notebook prototype

`test.ipynb` contains notebook cells that define the same functions inline.

If you see a PyTorch import error mentioning `TORCH_LIBRARY` and `prims`, it usually means the kernel got into a corrupted state (often after installing/upgrading torch mid-session). The fix is:

- restart the notebook kernel
- rerun the cells from the top

---

## Design decisions (why it’s written this way)

### 1) Model + regex hybrid

- ML models capture context-heavy entities (names, orgs, ambiguous patterns).
- Regex captures structural entities (IPs, postcodes, many phones) with high precision.

The combined set is then reconciled by overlap resolution.

### 2) Greedy overlap resolution

This chooses **one best span** for any region of text.

It is fast and deterministic, and the priority table makes the “business rules” explicit.

### 3) Right-to-left replacement

This ensures offsets remain correct during replacement.

### 4) Deduping by normalized value

If the same email appears five times, you typically want the same tag so downstream systems can reason about “the same entity”.

The normalization rules are label-specific so that formatting differences don’t create separate tags.

---

## Practical extension points

If you want to evolve this system, the safest knobs are:

1. Add/adjust canonical labels in `constants.py`.
2. Expand `GLINER_TO_CANON` mapping.
3. Add regex patterns in `regex_extractors.py`.
4. Update `PRIORITY` to control how overlaps are resolved.
5. Add stricter validators (and optionally drop spans when validators fail, if that becomes a requirement).
6. Tighten/loosen `scoring._match_score` and `sim_threshold`.

---

## Known limitations and edge cases

- Overlap resolution is greedy: it does not attempt a globally optimal set; it follows priority rules.
- Email normalization in `normalize.py` is intentionally aggressive; it removes punctuation, which can cause two distinct emails to collide if they only differ by punctuation. This is a tradeoff for dedupe stability.
- The regex patterns are UK-centric (UK phones, UK postcodes). If your data is multi-country, you’ll want additional patterns.
- Validators currently only adjust confidence; they do not suppress invalid-looking entities.
- GLiNER model output format changes could require updating how predictions are parsed.

---

## Mental model / pseudocode summary

At a high level:

1. `gliner_preds = gliner.predict_entities(text, labels, threshold)`
2. `spans = canonize(gliner_preds)`
3. `spans += regex_spans(text)`
4. `spans = resolve_overlaps(spans, priority_table)`
5. `for span in spans: tag = stable_tag(label, normalize(label, span.original))`
6. `masked_text = replace_right_to_left(text, spans, tag)`
7. `return masked_text, mapping, scores, spans`
