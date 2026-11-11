"""Ranking correctness tests: order, deduplication, grouping, and weighting."""
from collections import defaultdict
import math
import random
import pytest
from ml_eval.analytics.ranking_advanced import (
    group_ndcg_at_k, deduplicate_candidates, topk_per_query, weighted_query_metric
)


def reference_ndcg(rows,k):
    by=defaultdict(list)
    for i,row in enumerate(rows): by[row["query_id"]].append((i,row))
    if not by:return 0.0
    def dcg(labels):
        return sum((2**label-1)/math.log2(i+2) for i,label in enumerate(labels))
    scores=[]
    for group in by.values():
        ranked=sorted(group,key=lambda pair:(-pair[1]["score"],pair[0]))
        actual=[row["relevance"] for _,row in ranked[:k]]
        ideal=sorted((row["relevance"] for _,row in group),reverse=True)[:k]
        denominator=dcg(ideal)
        scores.append(dcg(actual)/denominator if denominator else 0.0)
    return sum(scores)/len(scores)


@pytest.mark.parametrize("nq,k", [(1,1),(2,3),(4,5),(10,10)])
def test_ndcg_independent_reference(nq,k):
    rng=random.Random(10+nq)
    rows=[{"query_id":i%nq,"doc_id":i,"score":rng.random(),
           "relevance":rng.randrange(4)} for i in range(180)]
    rng.shuffle(rows)
    assert group_ndcg_at_k(rows,k) == pytest.approx(reference_ndcg(rows,k))


def test_ranked_metrics_include_queries_with_zero_relevance():
    rows=[{"query_id":"a","score":1.0,"relevance":3},
          {"query_id":"b","score":1.0,"relevance":0}]
    assert group_ndcg_at_k(rows,1)==pytest.approx(0.5)


def test_dedup_keeps_highest_score_and_original_tie_winner():
    rows=[
        {"query_id":"q","doc_id":"a","score":0.2,"tag":"first"},
        {"query_id":"q","doc_id":"b","score":0.6,"tag":"b"},
        {"query_id":"q","doc_id":"a","score":0.9,"tag":"best"},
        {"query_id":"q","doc_id":"a","score":0.9,"tag":"late-tie"},
    ]
    selected=deduplicate_candidates(rows)
    lookup={(r["query_id"],r["doc_id"]):r for r in selected}
    assert lookup[("q","a")]["tag"]=="best"
    assert lookup[("q","b")]["tag"]=="b"


def test_topk_per_query_does_not_mix_queries_and_is_stable():
    rows=[{"query_id":"b","doc_id":"b1","score":0.4},
          {"query_id":"a","doc_id":"a1","score":0.8},
          {"query_id":"b","doc_id":"b2","score":0.9},
          {"query_id":"a","doc_id":"a2","score":0.8},
          {"query_id":"b","doc_id":"b3","score":0.9}]
    result=topk_per_query(rows,2)
    assert [r["doc_id"] for r in result]==["b2","b3","a1","a2"]


def test_weighted_metric_aligns_by_query_id_not_map_insertion_order():
    metrics={"q2":0.9,"q1":0.1}
    weights={"q1":9.0,"q2":1.0}
    assert weighted_query_metric(metrics,weights)==pytest.approx(0.18)


def test_weighted_metric_does_not_mutate_inputs():
    metrics={"q1":0.1,"q2":0.9}
    weights={"q1":9.0,"q2":1.0}
    before=(dict(metrics),dict(weights))
    weighted_query_metric(metrics,weights)
    assert (metrics,weights)==before
