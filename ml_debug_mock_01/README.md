# ML Evaluation Engineering Mock — Algorithm & Performance

**Independent interview-training project.** Not an official Anthropic / OpenAI / Resolution question. Built as a multi-module Python code-reading, debugging, algorithm and performance exercise.

## Metric contract (important)

- `samples`: requested evaluation population
- `outputs`: model predictions actually returned
- `EvalResult.n`: **number of returned predictions evaluated**
- `EvalResult.score`: correct / returned predictions (0.0 when no outputs)
- `per_sample`: one entry per returned prediction, keyed by its sample ID
- Request **coverage** is a separate metric: unique valid returned IDs / requested IDs. It must not silently change the meaning of accuracy.
- Duplicate or unknown output IDs are malformed provider data. Whether to raise, skip or report a separate error is a product-contract decision; document the policy before implementation.

A partial response is not automatically three incorrect answers when only one prediction was produced. The previous test that forced `n == len(samples)` was incorrect under this contract and has been fixed.

## Exercise A — Evaluation correctness (15–25 min)

Review `ml_eval/parsing.py`, `ml_eval/scoring.py`, and `ml_eval/pipeline.py`. Determine whether the observed metrics respect the above contract. Inspect malformed answers, out-of-order results, missing outputs and ID validation. Add meaningful tests rather than changing semantics just to make a test pass.

```bash
python -m pytest tests/test_regression.py -q
```

## Exercise B — Data split / leakage analysis (25–35 min)

Read `ml_eval/analytics/sampling.py` and `ml_eval/analytics/temporal.py`.

The evaluation stack must support reproducible random sampling, per-label stratification, grouped folds, and time-based train/test separation. Consider the hard case where multiple events belong to the same user, and identify which split contracts can and cannot simultaneously be guaranteed. Your task is to produce an explicit split specification, expose pitfalls with tests, and implement any new split behavior under a **new API** rather than silently changing existing event-scoped functions.

Questions for code review: balancing under tiny classes; group-wise leakage; crossing temporal boundaries; reproducibility independent of global RNG; complexity and memory consumption for large datasets.

```bash
python -m pytest tests/test_algorithm_contracts.py -q -k "split or group or time"
```

## Exercise C — Ranking algorithm and optimization (30–45 min)

Read `ml_eval/analytics/ranking.py` and the reference checks in `tests/test_algorithm_contracts.py`.

- Verify NDCG@K, AP, reciprocal rank and tie behavior with an independent small reference implementation.
- Analyze `ranked_indices` and `pairwise_accuracy` as N grows; compare algorithmic complexity.
- Design an exact, faster ranking implementation where appropriate, preserving tie semantics, rank correctness and public API behavior.
- Explain cases where using a heap for top-K does or does not actually save time; measure rather than assuming.
- Avoid optimizing a function by changing its mathematical definition.

```bash
python -m pytest tests/test_algorithm_contracts.py -q -k "ranking or ndcg or average_precision"
python benchmarks/algorithm_baseline.py
```

The benchmark is diagnostic, not a flaky pass/fail test. Record baseline, complexity, improved timing, correctness, and memory cost.

## Full suite

```bash
python -m pip install pytest
python -m pytest tests/test_regression.py tests/test_algorithm_contracts.py -q
python -m pytest tests/ -q
```

Tests are **not** proof that every algorithm in this large generated corpus is correct. Report additional defects and unintended mismatches rather than blindly weakening assertions.

## Timing

Treat the three exercises as separate 45–60 minute rounds in one large codebase. The ~10k+ lines come primarily from deterministic test corpora; line count is not a proxy for difficulty. You should focus on reading existing APIs, designing sound tests, and making nontrivial improvements without hidden semantic changes.

No answer key is committed to this branch.
