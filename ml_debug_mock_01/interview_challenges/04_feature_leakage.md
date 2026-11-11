# Hard Round 04 — Point-in-Time Features and Leakage

This is an **independently designed** ML debugging / performance exercise, not an official interview question.

Your pipeline serves features produced by delayed, out-of-order upstream events. Offline evaluation has unexpectedly improved, but online performance has regressed.

## Contract

A historical feature may influence a prediction only if its **event timestamp and ingestion timestamp** are both at or before the inference timestamp. For equal eligible event timestamps, prefer the last input event. Predictions with no available features must carry None. Labels cannot be observed before their availability time. A rolling CTR for an impression excludes the impression's own label.

## Candidate tasks

- Inspect `ml_eval/analytics/point_in_time.py`; locate silent feature leakage and aggregation bugs.
- Add tests for late-arriving events, duplicate timestamps, out-of-order inputs, missing entities and label delay.
- Investigate complexity of as-of join: naive O(I × F) versus indexing/search strategies with O((I+F) log F) sorting and efficient lookup; be careful because two timestamp constraints complicate binary search.
- Evaluate 100K impressions / 1M features for runtime and peak memory while preserving exact outputs.
- Explain how to prevent test set information from leaking into feature construction.
- Distinguish correctness from offline metric improvements caused by leaked labels or future features.

Run:

```bash
python -m pytest tests/test_point_in_time_hard.py -vv
```

Do not adjust correctness tests to accept leakage. Performance comparisons must use the same inputs and semantics. No external APIs or GPU required.
