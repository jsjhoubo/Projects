# Hard Round 03 — Ranking Quality + Offline Evaluation

Independent senior/staff-level ML algorithm debugging exercise, **not an official Anthropic/OpenAI problem**. Do not expect every numerical difference to be a software bug: first determine the evaluation contract and statistical meaning.

## Candidate assignment

You inherited ranking evaluation code used to compare candidate models. Several metrics disagree between offline reports and hand-verified examples, especially with ties, zero-positive queries, class imbalance, missing feedback, and logged exposure propensities.

1. Read `ml_eval/analytics/quality_advanced.py` and the related reference code in `ranking_advanced.py`.
2. Identify and fix algorithmic correctness defects. Do not change tests just to make them pass.
3. Validate NDCG@K, MAP@K, MRR@K on a **query macro-average**, not an accidental impression micro-average.
4. Preserve deterministic tie behavior; explain why score sorting order matters.
5. Diagnose SNIPS estimator assumptions, clipping, invalid propensities, missing/unclicked events, and why offline IPS does not establish unbiased online lift without a valid logging policy.
6. Check cohort-level metrics with overlapping query IDs, quality gaps, and edge cases.
7. Optimize hot paths at scale without silently changing metric semantics.

## Test commands

```bash
python -m pytest tests/test_ranking_quality_hard.py tests/test_ranking_advanced.py -vv
python -m benchmarks.ranking_quality_stress --rows 100000 --queries 500
python -m benchmarks.ranking_quality_stress --rows 1000000 --queries 500 --skew
```

The stress script is diagnostic, not a test with unstable wall-time thresholds. Record throughput, peak memory if available, and before/after outputs. For a runtime change, explain asymptotic complexity, grouping memory cost, tie correctness, and CPU overhead.

**Acceptance:** all numerical contracts pass; no unexpected metric regressions; optimized code demonstrably improves relevant workloads; original APIs remain compatible.
