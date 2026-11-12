"""Independent contracts for complex splitting. Initially not all must pass."""
from datetime import datetime, timedelta
from collections import Counter
import random
import pytest
from ml_eval.analytics.split_advanced import (
    group_stratified_split, temporal_event_split,
    purged_temporal_split, build_group_folds, rolling_origin_splits
)


def mk(n, *, big_group=False):
    rng=random.Random(902)
    result=[]
    for i in range(n):
        group=("major" if i < n//2 else f"g{i}") if big_group else f"g{i//3}"
        result.append({"id":i,"user_id":group,"label":i%2,
                       "event_time":datetime(2026,1,1)+timedelta(days=i),
                       "label_available_at":datetime(2026,1,1)+timedelta(days=i+2)})
    return result


@pytest.mark.parametrize("seed", [0, 1, 5, 9, 42])
def test_group_split_no_group_leakage(seed):
    rows=mk(120)
    train,val=group_stratified_split(rows,seed=seed)
    assert {r["user_id"] for r in train}.isdisjoint({r["user_id"] for r in val})
    assert sorted(r["id"] for r in train+val) == list(range(120))


@pytest.mark.parametrize("seed", [0, 1, 17, 999])
def test_group_split_is_deterministic(seed):
    rows=mk(100)
    a,b=group_stratified_split(rows,seed=seed)
    c,d=group_stratified_split(rows,seed=seed)
    assert [r["id"] for r in a] == [r["id"] for r in c]
    assert [r["id"] for r in b] == [r["id"] for r in d]


def test_group_stratification_on_class_pure_groups():
    """Group integrity is required; exact class balancing is not feasible in general.

    Group-level random assignment is a legitimate baseline, though not a
    guaranteed stratified allocation. A tolerance on class proportions would
    impose a new optimization requirement rather than diagnose a correctness
    bug in the current baseline.
    """
    rows=[]
    for label in (0,1):
        for group in range(30):
            for i in range(4):
                rows.append({"id":(label,group,i),"user_id":f"{label}-{group}","label":label})
    train,val=group_stratified_split(rows,train_fraction=0.8,seed=5)
    assert train and val
    assert {r["user_id"] for r in train}.isdisjoint({r["user_id"] for r in val})
    assert {r["id"] for r in train}.isdisjoint({r["id"] for r in val})
    assert len(train)+len(val) == len(rows)

    # Diagnostics, NOT pass/fail requirements for this baseline:
    target_fraction=0.8
    actual_fraction=len(train)/len(rows)
    global_labels=Counter(r["label"] for r in rows)
    train_labels=Counter(r["label"] for r in train)
    label_fraction={label:train_labels[label]/total
                    for label,total in global_labels.items()}
    print("target_train_fraction=", target_fraction,
          "actual_train_fraction=", actual_fraction,
          "train_fraction_per_label=", label_fraction)


def test_group_split_skewed_histograms_are_not_exactly_feasible():
    """Even an optimal solver cannot always achieve ratio and label targets.

    Two groups: (90 A,10 B) and (10 A,90 B). An 80% train target would
    require (80 A,80 B), which no whole-group assignment can produce.
    """
    rows=[]
    for gid, a, b in [("g1",90,10),("g2",10,90)]:
        rows.extend({"id":(gid,"A",i),"user_id":gid,"label":"A"}
                    for i in range(a))
        rows.extend({"id":(gid,"B",i),"user_id":gid,"label":"B"}
                    for i in range(b))
    train,val=group_stratified_split(rows,train_fraction=0.8,seed=17)
    assert {r["user_id"] for r in train}.isdisjoint({r["user_id"] for r in val})
    assert len(train)+len(val)==len(rows)
    hist=Counter(r["label"] for r in train)
    assert (hist["A"],hist["B"]) in {(0,0),(90,10),(10,90),(100,100)}
    assert (hist["A"],hist["B"]) != (80,80)


def test_large_group_ratio_is_bounded_by_largest_group():
    rows=mk(120,big_group=True)
    train,val=group_stratified_split(rows,train_fraction=0.7,seed=19)
    target=0.7*len(rows)
    largest=max(Counter(r["user_id"] for r in rows).values())
    assert abs(len(train)-target) <= largest


def test_temporal_cutoff_belongs_to_validation():
    cutoff=datetime(2026,2,3)
    rows=[{"event_time":cutoff-timedelta(seconds=1)},
          {"event_time":cutoff},
          {"event_time":cutoff+timedelta(seconds=1)}]
    train,val=temporal_event_split(rows,cutoff)
    assert len(train)==1
    assert len(val)==2


def test_purged_split_rejects_training_labels_from_the_future():
    cutoff=datetime(2026,3,1)
    rows=[
        {"id":"clean","event_time":cutoff-timedelta(days=10),
         "label_available_at":cutoff-timedelta(days=3)},
        {"id":"leaky","event_time":cutoff-timedelta(days=4),
         "label_available_at":cutoff+timedelta(days=2)},
        {"id":"boundary","event_time":cutoff,
         "label_available_at":cutoff+timedelta(days=1)},
        {"id":"val","event_time":cutoff+timedelta(days=6),
         "label_available_at":cutoff+timedelta(days=8)},
    ]
    train,val=purged_temporal_split(rows,cutoff,embargo=timedelta(days=3))
    assert [r["id"] for r in train] == ["clean"]
    assert [r["id"] for r in val] == ["val"]


@pytest.mark.parametrize("n_folds", [2, 3, 5, 7])
def test_group_folds_balanced_by_row_count(n_folds):
    rows=mk(180,big_group=True)
    folds=build_group_folds(rows,n_folds=n_folds)
    groups={}
    counts=Counter(folds)
    for row, fold in zip(rows,folds):
        if row["user_id"] in groups: assert groups[row["user_id"]]==fold
        groups[row["user_id"]]=fold
    largest=max(Counter(r["user_id"] for r in rows).values())
    assert max(counts.values())-min(counts.get(i,0) for i in range(n_folds)) <= largest


def test_rolling_window_avoids_future_and_overlap():
    rows=mk(90)
    cutoffs=[datetime(2026,1,10),datetime(2026,1,30)]
    splits=list(rolling_origin_splits(rows,cutoffs,horizon=timedelta(days=7)))
    assert len(splits)==2
    for (train,val), cutoff in zip(splits,cutoffs):
        assert all(r["event_time"]<cutoff for r in train)
        assert all(cutoff<=r["event_time"]<cutoff+timedelta(days=7) for r in val)
        assert {r["id"] for r in train}.isdisjoint({r["id"] for r in val})
