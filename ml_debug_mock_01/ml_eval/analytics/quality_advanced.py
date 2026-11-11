"""Offline recommendation/ranking quality evaluation.

Contracts are in interview_challenges/03_ranking_quality.md. Treat unexpected
metric shifts as model or evaluation bugs, not as an invitation to change tests.
"""
from __future__ import annotations
from collections import defaultdict
import math


def group_map_at_k(rows, k=10, *, query_key="query_id",
                   score_key="score", label_key="clicked"):
    """Macro-average AP@k over nonempty queries, counting zero-positive as 0.

    Only positive labels count as relevant. AP@k is divided by
    min(number of positives in query, k). Stable input order for ties.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        groups[row[query_key]].append((i,row))
    if not groups:
        return 0.0
    total=0.0
    for entries in groups.values():
        ranked=sorted(entries,key=lambda r:(-r[1][score_key],r[0]))[:k]
        positives=sum(int(row[label_key]>0) for _,row in entries)
        if positives == 0:
            continue
        hit=0
        score=0.0
        for rank, (_, row) in enumerate(ranked,1):
            if row[label_key]:
                hit += 1
                score += hit/rank
        total+=score/min(positives,k)
    return total/len(groups)


def group_mrr_at_k(rows,k=10,*,query_key="query_id",
                   score_key="score",label_key="clicked"):
    """Mean reciprocal first relevant rank, with zero-positive queries as 0."""
    groups=defaultdict(list)
    for idx,row in enumerate(rows):
        groups[row[query_key]].append((idx,row))
    if k<=0:
        raise ValueError("k must be positive")
    if not groups:return 0.0
    result=0.0
    for entries in groups.values():
        ranked=sorted(entries,key=lambda t:(-t[1][score_key],t[0]))[:k]
        for i,(_,row) in enumerate(ranked,1):
            if row[label_key]>0:
                result+=1/i
                break
    return result/len(groups)


def rank_weighted_ctr(rows, *, query_key="query_id", position_key="position",
                      click_key="clicked", propensity_key="propensity",
                      score_key="score", clip=20.0):
    """Self-normalized IPS click estimate over logged events.

    Missing / invalid propensity is a data-quality error. Each event is a
    logged impression, not a candidate that was never exposed.
    """
    if clip<=0:raise ValueError("positive clip required")
    numerator=0.0
    denominator=0.0
    for row in rows:
        p=float(row[propensity_key])
        if not 0<p<=1:raise ValueError("propensity must lie in (0,1]")
        w=min(1/p,clip)
        numerator+=w*float(row[click_key])
        denominator+=w
    return numerator/denominator if denominator else 0.0


def rank_lift_per_query(rows, *, query_key="query_id",
                        baseline_key="baseline", challenger_key="challenger",
                        label_key="relevance",k=10):
    """Macro-average NDCG@k(challenger) - NDCG@k(baseline).

    A query with no relevant items scores zero for both policies.
    Exact equal scores are stable in original input order.
    """
    from .ranking_advanced import group_ndcg_at_k
    rows=list(rows)
    a=[{**r,"score":r[baseline_key]} for r in rows]
    b=[{**r,"score":r[challenger_key]} for r in rows]
    return group_ndcg_at_k(b,k)-group_ndcg_at_k(a,k)


def ranking_quality_by_slice(rows,slice_key,k=10):
    """Compute macro NDCG@k for each cohort, respecting query boundaries.

    A query may appear in several slices and counts once within each slice.
    Missing slice values belong to the explicit None slice.
    """
    from .ranking_advanced import group_ndcg_at_k
    cohorts=defaultdict(list)
    for row in rows:
        cohorts[row.get(slice_key)].append(row)
    return {key:group_ndcg_at_k(group,k) for key,group in cohorts.items()}
