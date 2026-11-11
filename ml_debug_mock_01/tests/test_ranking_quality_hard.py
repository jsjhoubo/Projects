"""Ranking-quality contracts based on small, independently computed oracles.

Failing tests require debugging the algorithm, not weakening the metric.
"""
from collections import defaultdict
import math
import random
import pytest
from ml_eval.analytics.quality_advanced import (
    group_map_at_k,group_mrr_at_k,rank_weighted_ctr,
    rank_lift_per_query,ranking_quality_by_slice
)


def oracle_ap(rows,k):
    groups=defaultdict(list)
    for idx,row in enumerate(rows):
        groups[row["query_id"]].append((idx,row))
    if not groups:return 0.
    totals=[]
    for group in groups.values():
        positives=sum(row["clicked"]>0 for _,row in group)
        order=sorted(group,key=lambda pair:(-pair[1]["score"],pair[0]))[:k]
        hits=0
        numerator=0.0
        for rank,(_,row) in enumerate(order,1):
            if row["clicked"]>0:
                hits+=1
                numerator+=hits/rank
        totals.append(numerator/min(k,positives) if positives else 0.0)
    return sum(totals)/len(totals)


@pytest.mark.parametrize("k", [1,2,3,5,10,20])
@pytest.mark.parametrize("seed", [1,17,29,67])
def test_map_at_k_against_reference(seed,k):
    rng=random.Random(seed)
    rows=[{"query_id":f"q{i%5}","score":rng.choice([0.2,0.2,0.5,0.9]),
           "clicked":rng.randrange(2)} for i in range(73)]
    rng.shuffle(rows)
    assert group_map_at_k(rows,k)==pytest.approx(oracle_ap(rows,k),abs=1e-12)


def test_map_uses_correct_macro_query_weighting():
    rows=[
        {"query_id":"a","score":1.0,"clicked":1},
        {"query_id":"b","score":0.9,"clicked":0},
        {"query_id":"b","score":0.8,"clicked":1},
        {"query_id":"b","score":0.7,"clicked":0},
    ]
    assert group_map_at_k(rows,3)==pytest.approx((1+1/2)/2)


def test_mrr_zero_positive_query_contributes_zero():
    rows=[
        {"query_id":"a","score":1.0,"clicked":1},
        {"query_id":"b","score":1.0,"clicked":0},
        {"query_id":"b","score":0.5,"clicked":0},
    ]
    assert group_mrr_at_k(rows,5)==pytest.approx(0.5)


def test_map_duplicate_scores_have_deterministic_input_order():
    rows=[{"query_id":"q","score":0.5,"clicked":0},
          {"query_id":"q","score":0.5,"clicked":1}]
    assert group_map_at_k(rows,2)==pytest.approx(0.5)


@pytest.mark.parametrize("k", [1,2,3])
def test_map_denominator_caps_at_k(k):
    rows=[{"query_id":"q","score":1-i*.1,"clicked":1} for i in range(5)]
    assert group_map_at_k(rows,k)==pytest.approx(1.0)


def test_snips_equals_mean_click_when_propensity_is_one():
    rows=[{"clicked":1,"propensity":1,"position":i,"query_id":"q","score":1}
          for i in range(3)]
    rows.extend({"clicked":0,"propensity":1,"position":i,"query_id":"q","score":1}
                for i in range(3,10))
    assert rank_weighted_ctr(rows)==pytest.approx(0.3)


def test_ips_weighting_is_not_raw_ctr():
    rows=[{"clicked":1,"propensity":0.1,"query_id":"a","position":0,"score":0.3},
          {"clicked":0,"propensity":1.0,"query_id":"b","position":0,"score":0.8}]
    assert rank_weighted_ctr(rows)==pytest.approx(10/11)
    assert rank_weighted_ctr(rows)!=0.5


@pytest.mark.parametrize("bad_p", [0,-1,1.01,float("nan")])
def test_ips_rejects_invalid_propensities(bad_p):
    with pytest.raises(ValueError):
        rank_weighted_ctr([{"clicked":1,"propensity":bad_p,"query_id":"a",
                            "position":0,"score":0.5}])


def test_ips_clipping_is_explicit():
    rows=[{"clicked":1,"propensity":0.001,"query_id":"a","position":0,"score":0.5},
          {"clicked":0,"propensity":1.0,"query_id":"b","position":0,"score":0.5}]
    assert rank_weighted_ctr(rows,clip=5)==pytest.approx(5/6)


def test_identical_rankers_have_zero_lift():
    rows=[{"query_id":i%3,"relevance":i%4,"baseline":i/50,
           "challenger":i/50} for i in range(50)]
    assert rank_lift_per_query(rows,k=5)==pytest.approx(0.0)


def test_better_ranker_has_positive_ndcg_lift():
    rows=[
        {"query_id":"q","relevance":3,"baseline":0.1,"challenger":1.0},
        {"query_id":"q","relevance":0,"baseline":1.0,"challenger":0.1},
        {"query_id":"q","relevance":1,"baseline":0.3,"challenger":0.5}]
    assert rank_lift_per_query(rows,k=3)>0


def test_slice_evaluation_respects_query_boundaries():
    rows=[
        {"query_id":"q1","country":"us","score":1.0,"relevance":3},
        {"query_id":"q1","country":"us","score":0.0,"relevance":0},
        {"query_id":"q2","country":"ca","score":1.0,"relevance":0},
        {"query_id":"q2","country":"ca","score":0.5,"relevance":2},
    ]
    metrics=ranking_quality_by_slice(rows,"country",k=2)
    assert metrics["us"]==pytest.approx(1.0)
    assert metrics["ca"]<1.0


@pytest.mark.parametrize("nq", [1,3,10])
def test_ranker_lift_zero_on_identity_with_interleaved_queries(nq):
    rng=random.Random(nq)
    rows=[{"query_id":i%nq,"relevance":rng.randrange(4),
           "baseline":rng.random()} for i in range(150)]
    for r in rows:r["challenger"]=r["baseline"]
    rng.shuffle(rows)
    assert rank_lift_per_query(rows,k=10)==pytest.approx(0)
