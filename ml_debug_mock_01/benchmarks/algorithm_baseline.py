"""CPU-only diagnostic benchmark for ranking algorithm analysis.

No pass/fail performance assertions: timings vary across laptops.
Use exact baselines before changing implementation.
"""
from __future__ import annotations
import random
import statistics
import time

from ml_eval.analytics.ranking import pairwise_accuracy, ranked_indices, ndcg


def measure(fn, repeats=3):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return result, statistics.median(values)


def run():
    rng = random.Random(2026)
    print("n | sort rank(s) | NDCG@20(s) | pairwise(s)")
    for n in [250, 500, 1000, 2000]:
        predictions = [rng.random() for _ in range(n)]
        relevance = [rng.randrange(4) for _ in range(n)]
        _, ranking_sec = measure(lambda: ranked_indices(predictions))
        _, ndcg_sec = measure(lambda: ndcg(predictions, relevance, k=20))
        _, pairwise_sec = measure(lambda: pairwise_accuracy(predictions, relevance), repeats=2)
        print(f"{n:4} | {ranking_sec:.6f} | {ndcg_sec:.6f} | {pairwise_sec:.6f}")


if __name__ == "__main__":
    run()
