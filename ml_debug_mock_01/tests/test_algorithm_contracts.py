"""Algorithm-focused contracts: numerical correctness and dataset leakage.

These are not performance timing tests. They define mathematical / data-splitting
behavior and can be run without API keys, GPU, or optional dependencies.
"""
import math
import pytest

from ml_eval.analytics import ranking, sampling, temporal


def reference_dcg(scores, relevance, k):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return sum((2 ** relevance[i] - 1) / math.log2(j + 2)
               for j, i in enumerate(order))


@pytest.mark.parametrize("k", [1, 2, 3, 4, 6])
def test_ndcg_matches_independent_reference(k):
    predictions = [0.4, 0.95, 0.4, 0.2, 0.8, 0.0]
    relevance = [2, 3, 1, 0, 2, 0]
    numerator = reference_dcg(predictions, relevance, k)
    denominator = reference_dcg(relevance, relevance, k)
    expected = numerator / denominator
    assert ranking.ndcg(predictions, relevance, k) == pytest.approx(expected)


def test_average_precision_uses_ranked_positions():
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [1, 0, 1, 0]
    expected = (1.0 + 2 / 3) / 2
    assert ranking.average_precision(scores, labels) == pytest.approx(expected)


def test_ranking_ties_are_stable_and_deterministic():
    assert ranking.ranked_indices([0.5, 0.5, 0.9, 0.5]) == [2, 0, 1, 3]


@pytest.mark.parametrize("size,ratio", [(0,0.5),(1,0.5),(101,0.2),(100,0.8)])
def test_random_split_partitions_without_duplication(size,ratio):
    items = list(range(size))
    left,right = sampling.stable_partition(items,ratio,seed=5)
    assert len(left) == int(size * ratio)
    assert sorted(left+right) == items
    assert not (set(left) & set(right))


def test_random_split_is_reproducible_without_global_random_side_effects():
    items = list(range(300))
    first = sampling.stable_partition(items,0.7,seed=17)
    second = sampling.stable_partition(items,0.7,seed=17)
    different = sampling.stable_partition(items,0.7,seed=18)
    assert first == second
    assert first != different


def test_group_folds_never_split_a_group_across_folds():
    group_ids = ["u1","u1","u2","u3","u3","u4","u5","u5"]
    assigned = sampling.grouped_fold(group_ids,n_folds=3,seed=5)
    assert len(assigned) == len(group_ids)
    assert set(assigned).issubset({0,1,2})
    seen = {}
    for group,fold in zip(group_ids,assigned):
        if group in seen:
            assert seen[group] == fold
        else:
            seen[group] = fold


def test_stratified_split_preserves_all_items_and_per_class_counts():
    ninety = 90
    items = [{"id":i,"label":i%3} for i in range(ninety)]
    left,right = sampling.stratified_partition(items,lambda r:r["label"],0.7,seed=7)
    assert sorted(r["id"] for r in left+right) == list(range(ninety))
    for label in range(3):
        assert sum(r["label"]==label for r in left) == int(30*0.7)


def test_time_split_respects_boundary_and_prevents_future_examples_in_train():
    events = [{"id":i,"t":(i%11)} for i in range(200)]
    train,test=temporal.split_before(events,"t",5)
    assert all(event["t"]<5 for event in train)
    assert all(event["t"]>=5 for event in test)
    assert len(train)+len(test)==len(events)


def test_group_temporal_split_needs_explicit_semantic_choice():
    """Simple temporal split is event-scoped, not group-scoped."""
    events=[{"user":"a","t":1},{"user":"a","t":10},
            {"user":"b","t":2},{"user":"b","t":11}]
    train,test=temporal.split_before(events,"t",5)
    assert {r["user"] for r in train} == {"a","b"}
    assert {r["user"] for r in test} == {"a","b"}
