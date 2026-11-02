# LEVEL 4 — Point-in-Time ML Feature Export (Silent Data Bug)

## Business context
RecordFlow now exports historical training examples for a model predicting business engagement. A training example is created for a business at a specific **label time**. Supplier snapshots are historical versions of the business profile. Training features must represent information that was available **as of the label time**, not information learned afterward. The export has produced plausible-looking rows but the offline model performs suspiciously better than online serving. Diagnose and fix the implementation.

## Your task
A starter implementation with intentionally incorrect behavior is supplied as `recordflow/feature_export.py`. **Fix that module**, preserving the public API below. This is a Debugging level, not a new end-to-end ingestion feature. Do not alter Levels 1–3, `BusinessRecord`, or test files.

```python
@dataclass(frozen=True)
class TrainingExample:
    business_id: str
    label_time: datetime
    city: str | None
    source: str | None
    record_age_hours: float | None


def build_training_examples(
    snapshots: Iterable[BusinessRecord],
    labels: Iterable[tuple[str, datetime]],
) -> list[TrainingExample]: ...
```

## Input contract
- `snapshots` is an iterable of **historical** `BusinessRecord` versions, not necessarily the current `MemoryBusinessStore`. Many snapshots may share `business_id`; records arrive in any order.
- `labels` is an iterable of `(business_id, label_time)` pairs; preserve its input order and emit exactly one example per pair. The same business may have multiple labels.
- `business_id` strings already use their canonical form; exact match. `label_time` and all `updated_at` values are timezone-aware `datetime` objects. Do not mutate inputs.
- Snapshot versions can have equal timestamps. If two or more versions for the *same business* have the same maximal eligible timestamp, **the last of those versions in the input iterable wins**.

## Required behavior
For each `(business_id, label_time)`:
1. Find the snapshot with that ID and the greatest `updated_at` **not later than** `label_time` (`updated_at <= label_time`). This is an **as-of join**. A snapshot newer than the label must never be used, even if it is the most recent snapshot overall.
2. If no eligible snapshot exists, return `TrainingExample(business_id, label_time, None, None, None)`.
3. Otherwise return the chosen snapshot's `city` and `source`, and compute `record_age_hours = (label_time - chosen.updated_at).total_seconds() / 3600.0`. Age must never be negative.
4. Preserve label order and do not drop duplicates. Do not add fields or change `TrainingExample`.

## Worked example
Snapshots in this input order:

```text
BusinessRecord('a', 'Cafe', 'Boston', 2026-01-03 00:00 UTC, 'beacon')
BusinessRecord('a', 'Cafe', 'NYC',    2026-01-01 00:00 UTC, 'atlas')
BusinessRecord('b', 'Deli', 'Philly', 2026-01-02 00:00 UTC, 'atlas')
```

Labels in this order:

```text
('a', 2026-01-02 12:00 UTC)
('b', 2026-01-01 12:00 UTC)
('a', 2026-01-04 00:00 UTC)
```

Expected output in the *same label order*:

```text
TrainingExample('a', Jan 02 12:00, city='NYC',    source='atlas',  record_age_hours=36.0)
TrainingExample('b', Jan 01 12:00, city=None,     source=None,     record_age_hours=None)
TrainingExample('a', Jan 04 00:00, city='Boston', source='beacon', record_age_hours=24.0)
```

## Constraints and acceptance
- Up to 100,000 snapshots and 100,000 labels. Avoid scanning every snapshot for every label; target `O((S+L) log (S+L))` time and `O(S+L)` auxiliary space or better.
- Only Python standard library is needed. You can add helper functions or data structures within `feature_export.py`.
- The new file intentionally contains *silent* bugs: it may return rows without throwing any exception. Find them using spec, tests, and small examples.
- Run `python -m pytest tests/test_level4.py -v`. Then run all earlier levels to confirm no regressions.
- The test file is independent of the implementation. No additional hidden business requirements are introduced.
