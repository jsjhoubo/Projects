"""Large-scale, multi-query ranking evaluation and candidate reranking."""
from __future__ import annotations
from collections import defaultdict
import math


def group_ndcg_at_k(rows, k=10, *, query_key="query_id",
                    score_key="score", relevance_key="relevance"):
    """Macro-average NDCG@K across queries with at least one record.

    Rows may be arbitrarily interleaved. Ties must be deterministic using
    original row order. A query with zero ideal DCG contributes zero.
    """
    if k < 1:
        raise ValueError("k must be positive")
    by_query = defaultdict(list)
    for ix, row in enumerate(rows):
        by_query[row[query_key]].append((ix, row))
    if not by_query:
        return 0.0
    total = 0.0
    for query, group in by_query.items():
        ordered = sorted(group, key=lambda z: (-z[1][score_key], z[0]))
        labels = [pair[1][relevance_key] for pair in ordered[:k]]
        all_labels = sorted((pair[1][relevance_key] for pair in group), reverse=True)[:k]
        def dcg(values):
            return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(values))
        ideal = dcg(all_labels)
        total += dcg(labels) / ideal if ideal else 0.0
    return total / len(by_query)


def deduplicate_candidates(rows, *, query_key="query_id", doc_key="doc_id",
                           score_key="score"):
    """One result per query/doc; retain highest-scoring instance.

    For a tie, retain the earliest input row. Preserve the chosen rows'
    original relative order in returned output.
    """
    best = {}
    for row in rows:
        key = (row[query_key], row[doc_key])
        if key not in best:
            best[key] = row
        elif row[score_key] >= best[key][score_key]:
            best[key] = row
    return list(best.values())


def topk_per_query(rows, k, *, query_key="query_id", score_key="score"):
    """Rank each query independently, stable under score ties.

    Output groups are ordered by the first appearance of the query, then by
    decreasing score, with input order as the deterministic tie-breaker.
    """
    if k < 0:
        raise ValueError("k cannot be negative")
    by_query = defaultdict(list)
    for i, row in enumerate(rows):
        by_query[row[query_key]].append((i, row))
    ranked = []
    for query, group in by_query.items():
        group.sort(key=lambda pair: (-pair[1][score_key], pair[0]))
        ranked.extend(row for _, row in group[:k])
    return ranked


def weighted_query_metric(query_metrics, query_weights):
    """Return a properly weighted mean, matching query IDs rather than order."""
    if set(query_metrics) != set(query_weights):
        raise ValueError("query IDs must match")
    if any(w < 0 for w in query_weights.values()):
        raise ValueError("weights cannot be negative")
    denominator = sum(query_weights.values())
    if denominator <= 0:
        return 0.0
    return sum(score * weight for score, weight in
               zip(query_metrics.values(), query_weights.values())) / denominator
