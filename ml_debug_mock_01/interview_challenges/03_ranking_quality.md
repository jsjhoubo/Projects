# Hard Round 03 — Offline Ranking Evaluation & Statistical Debugging

Independent Senior/Staff-level ML debugging exercise; **not an official Anthropic/OpenAI question**.

**Scope:** investigate whether an offline ranking evaluation measures what its authors *think* it measures. This round is distinct from **Round 02**, which already covers NDCG/MAP/MRR ranking mechanics, deterministic ties, Top-K heap optimization, and ranking performance. **Do not redo those tasks.** The existing 52 passing ranking regression tests are a baseline, not proof of statistical validity.

## Scenario

An offline report claims a challenger ranking model improved engagement. The production A/B test is neutral or negative. Logged impressions have positions, clicks, exposure propensities, query IDs, and cohort attributes; feedback can be delayed or absent. You inherited `ml_eval/analytics/quality_advanced.py`. Your objective is to distinguish a bug in the code from an invalid statistical assumption, a biased evaluation dataset, or a mismatch in the target estimand.

## Task A — IPS / SNIPS and selection bias (30–45 min)

Focus on `rank_weighted_ctr()` in `ml_eval/analytics/quality_advanced.py`.

1. **Define the estimand first.** Is this estimating observed CTR, a position-debiased logged CTR, or the counterfactual value of a *different ranking policy*? Explain which claims the available fields can support, and which require logging-policy/target-policy probabilities or more information.
2. Read the weight construction and clipping contract. Audit zero, negative, >1, NaN and missing propensities; empty input; nonbinary/invalid clicks; zero denominator; and numerical stability under very small probabilities. Build minimal counterexamples.
3. Analyze IPS versus self-normalized IPS: bias/variance trade-offs, support/positivity assumptions, effective sample size, and the consequences of clipping.
4. Explain why the estimator may be misleading when only exposed items are logged, positions affect clicks, or a target policy chooses candidates that had no chance of exposure.
5. Propose additional focused regression tests that check **mathematical definitions** rather than arbitrary expected constants. Make surgical fixes where the agreed contract is violated; do not change metrics just to pass assertions.

**Important:** `rank_weighted_ctr` currently computes a clipped, self-normalized weighted mean of observed clicks. Its name/available fields alone do not prove that it is an unbiased *off-policy* estimator. If true OPE is needed, state the missing data and specification instead of inventing an estimator.

## Task B — Missing / delayed feedback and evaluation leakage (25–40 min)

You discover that some impressions are missing click outcomes, and others receive labels after the evaluation cutoff.

1. Distinguish **observed negative**, **unobserved / missing**, and **not yet mature** labels. Decide which are eligible for an evaluation at time T, and explain the resulting selection bias.
2. Design tests for late-arriving conversions, truncated feedback windows, duplicated impression IDs, repeated exposure of the same item, and event-time versus ingestion-time.
3. Demonstrate how exclusion, implicit zero-filling, or premature inclusion can alter offline CTR or policy-comparison conclusions. Do not implement a new dataset pipeline unless the diagnosis requires it.

## Task C — Cohorts, query composition and metric interpretation (25–40 min)

Inspect `ranking_quality_by_slice()` and `rank_lift_per_query()`.

1. Verify that **query IDs can repeat across slices** without accidentally combining groups, and that a cohort's query-level metric is computed on the intended population.
2. Examine heterogeneous query sizes and cohort imbalance: query-macro versus impression-micro averaging, changes in query mix, overlapping slices, and small or zero-positive cohorts.
3. Specify how you would compare baseline and challenger on *paired query traffic*. Explain why comparing two populations with different query distributions may produce misleading lift.
4. Construct a counterexample showing a healthy global metric with a deteriorating important cohort, or vice versa. Report the aggregation rule explicitly.

## Task D — Offline / online disagreement: root-cause review (20–30 min)

Given a positive offline lift and a neutral/negative online A/B outcome, produce a **prioritized debugging plan**. Consider selection and position bias, policy-support violations, label delay, point-in-time leakage, candidate-set differences, calibration, model/feature freshness, and offline versus online metric definitions.

For each hypothesis, state: (a) what observation would support it; (b) one diagnostic or replay experiment; (c) the main confounder; (d) whether a code change is actually justified.

**Do not** reimplement NDCG, MAP, MRR, Top-K or a heap as part of this round. Reuse the already-tested Round 02 implementation.

## Existing regression commands

```bash
# Round 03's existing tests (a starting baseline)
python -m pytest tests/test_ranking_quality_hard.py -vv

# Optional dependency regression; no need to redo Round 02
python -m pytest tests/test_ranking_advanced.py -q
```

**Test coverage limitation:** the current tests cover parts of the numerical contract (e.g., weighting, clipping, slices, lift) but do **not** establish unbiased off-policy evaluation or cover every missing-feedback case. New focused tests and diagnostic analysis are part of this task.

The `benchmarks/ranking_quality_stress.py` workload belongs to **Round 02's performance track**. It is intentionally **not a completion requirement** for Round 03.

## Completion criteria

- Written, explicit estimand and logging/feedback assumptions.
- Correctness checks against small hand-calculated or independent reference examples.
- A clear account of IPS/SNIPS applicability, clipping and support limitations.
- A defensible evaluation policy for missing/delayed feedback and cohort aggregation.
- Minimal code fixes *only* where an established contract is violated, with targeted regression tests.
- A concrete diagnosis plan for offline-vs-online metric discrepancies.

This round is deliberately about **statistical correctness and ML debugging**, not another round of ranking data-structure optimization.
