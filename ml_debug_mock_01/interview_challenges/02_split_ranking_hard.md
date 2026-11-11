# Hard Round 02 — Data Split + Learning-to-Rank

**Independent practice problem, not an official Anthropic/OpenAI problem.**
The candidate should not assume that all failures indicate one-line bugs. The expected tasks include algorithm design, changing asymptotic cost safely, and precise ML evaluation semantics.

## Task A: Dataset splitting and leakage (35–55 min)
Review `ml_eval/analytics/split_advanced.py` and its tests. There are multiple interacting constraints:
* Group integrity (one user cannot appear in train and validation).
* Class stratification and bounded fractional split error.
* Reproducibility without manipulating global random state.
* Strict time boundary and delayed-label eligibility.
* Purge and embargo semantics.
* Rolling-origin evaluation with half-open validation windows.
* Group K-fold balance by number of **records**, not number of groups.

Do not silently change one API to satisfy an incompatible contract. Explain feasibility and tradeoffs first. Write a counterexample for every fix.

```bash
python -m pytest tests/test_split_advanced.py -vv
```

## Task B: Multi-query ranking and correctness (35–55 min)
Review `ml_eval/analytics/ranking_advanced.py`:
* Query-level NDCG@K macro averaging, even under zero-positive queries.
* Deterministic tie-breaking.
* Candidate deduplication under repeated query/document IDs.
* Per-query Top-K with interleaved requests.
* Weighted metrics keyed by identity rather than mapping order.

Tests encode observable contracts, not implementation details. Preserve the public API and explain complexity.

```bash
python -m pytest tests/test_ranking_advanced.py -vv
```

## Task C: Performance engineering (45–60 min)
Benchmark the same implementations on 100K–1M synthetic candidates and events. Focus on end-to-end wall time, peak memory, regression correctness and the true computational bottleneck, not merely algorithm names. Consider sorting versus heap-based selection, stable tie handling, copying costs, and worst-case skew.

Acceptance requires: a before/after measurement under the same workload, a complexity argument, and unchanged outputs on correctness tests. Avoid machine-specific absolute time thresholds.

## Ground rules
- Do not modify assertions to make defects disappear.
- A source test failure can reveal specification ambiguity; resolve the API contract first.
- Avoid external network requests, GPU, and paid services.
- This task is intended to be harder than the previous ML evaluator parsing task.
