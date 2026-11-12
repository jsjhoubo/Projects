"""Tree split correctness contracts. Some tests intentionally detect algorithm defects."""
import math
import random
import pytest

from ml_eval.analytics.tree_split import (
    best_split_exact, best_split_histogram, split_rows,
)


def reference_gain(grad, hess, left, right, lam, gamma):
    def score(ids):
        g=sum(grad[i] for i in ids)
        h=sum(hess[i] for i in ids)
        return g*g/(h+lam)
    all_ids=left+right
    return .5*(score(left)+score(right)-score(all_ids))-gamma


def reference_search(rows, grad, hess, *, reg_lambda=1.0, gamma=0.0,
                     min_child_samples=1, min_child_weight=0.0):
    """Small exhaustive oracle: evaluate decisions on ORIGINAL feature units."""
    if not rows:
        return None
    candidates=[]
    for f in range(len(rows[0])):
        distinct=sorted({row[f] for row in rows if math.isfinite(row[f])})
        for threshold in distinct[:-1]:
            for missing_left in (True,False):
                left=[i for i,row in enumerate(rows)
                      if (missing_left if math.isnan(row[f]) else row[f]<=threshold)]
                right=[i for i in range(len(rows)) if i not in set(left)]
                if min(len(left),len(right))<min_child_samples:
                    continue
                if min(sum(hess[i] for i in left),sum(hess[i] for i in right))<min_child_weight:
                    continue
                gain=reference_gain(grad,hess,left,right,reg_lambda,gamma)
                if gain>0:
                    candidates.append((gain,f,threshold,missing_left,left,right))
    if not candidates:
        return None
    candidates.sort(key=lambda z:(-z[0],z[1],z[2],not z[3]))
    return candidates[0]


def assert_split_consistent(rows,grad,hess,result,**kw):
    assert result is not None
    left,right=split_rows(rows,result)
    assert not set(left)&set(right)
    assert sorted(left+right)==list(range(len(rows)))
    assert len(left)==result.left_count
    assert len(right)==result.right_count
    assert result.gain==pytest.approx(
        reference_gain(grad,hess,left,right,kw.get("reg_lambda",1.0),
                       kw.get("gamma",0.0)),abs=1e-9)


@pytest.mark.parametrize("seed",[1,3,5,11])
def test_exact_search_matches_exhaustive_oracle(seed):
    rng=random.Random(seed)
    rows=[[rng.choice([0.,1.,2.,3.,float("nan")]),rng.choice([1.,2.,3.])]
          for _ in range(25)]
    grad=[rng.uniform(-2.,2.) for _ in rows]
    hess=[rng.uniform(.1,2.) for _ in rows]
    expected=reference_search(rows,grad,hess)
    actual=best_split_exact(rows,grad,hess)
    assert (actual is None)==(expected is None)
    if expected:
        gain,f,threshold,missing_left,_,_=expected
        assert actual.feature==f
        assert actual.threshold==threshold
        assert actual.missing_left==missing_left
        assert actual.gain==pytest.approx(gain,abs=1e-9)


def test_duplicate_feature_values_not_split_between_children():
    rows=[[1.],[1.],[1.],[3.],[3.]]
    grad=[-2.,-2.,-2.,3.,3.]
    hess=[1.]*5
    split=best_split_exact(rows,grad,hess)
    assert_split_consistent(rows,grad,hess,split)
    left,right=split_rows(rows,split)
    assert {rows[i][0] for i in left}.isdisjoint({rows[i][0] for i in right})


def test_missing_direction_is_part_of_split_decision():
    rows=[[0.],[1.],[2.],[float("nan")]]
    grad=[-4.,4.,4.,-4.]
    hess=[1.]*4
    split=best_split_exact(rows,grad,hess)
    assert_split_consistent(rows,grad,hess,split)


@pytest.mark.parametrize("minimum",[1,2,3])
def test_min_child_sample_constraint(minimum):
    rows=[[float(i)] for i in range(6)]
    grad=[-5.,-4.,-3.,3.,4.,5.]
    hess=[1.]*6
    split=best_split_exact(rows,grad,hess,min_child_samples=minimum)
    if split:
        left,right=split_rows(rows,split)
        assert min(len(left),len(right))>=minimum


def test_no_positive_gain_returns_no_split():
    rows=[[0.],[1.],[2.]]
    grad=[0.,0.,0.]
    hess=[1.,1.,1.]
    assert best_split_exact(rows,grad,hess) is None


def test_histogram_split_returns_threshold_in_original_feature_units():
    rows=[[100.],[200.],[300.],[400.],[500.],[600.]]
    grad=[-4.,-4.,-4.,4.,4.,4.]
    hess=[1.]*6
    split=best_split_histogram(rows,grad,hess,max_bins=4)
    assert_split_consistent(rows,grad,hess,split)


@pytest.mark.parametrize("seed",[7,13,23])
def test_histogram_partition_and_reported_gain_agree(seed):
    rng=random.Random(seed)
    rows=[[rng.uniform(30.,70.)] for _ in range(50)]
    grad=[rng.uniform(-3.,3.) for _ in rows]
    hess=[rng.uniform(.3,1.5) for _ in rows]
    split=best_split_histogram(rows,grad,hess,max_bins=8)
    if split:
        assert_split_consistent(rows,grad,hess,split)


def test_histogram_respects_child_sample_constraints_on_original_rows():
    rows=[[float(i)*100] for i in range(20)]
    grad=[-4.]*8+[3.]*12
    hess=[1.]*20
    split=best_split_histogram(rows,grad,hess,max_bins=6,min_child_samples=5)
    if split:
        left,right=split_rows(rows,split)
        assert min(len(left),len(right))>=5


@pytest.mark.parametrize("invalid",[
    [float("inf")],[float("-inf")]
])
def test_exact_rejects_infinite_feature_values(invalid):
    with pytest.raises(ValueError):
        best_split_exact([[1.],invalid],[1.,-1.],[1.,1.])


def test_exact_does_not_modify_input():
    rows=[[3.],[1.],[2.],[float("nan")]]
    grad=[1.,-1.,2.,-2.]
    hess=[1.,1.,1.,1.]
    copied=[row[:] for row in rows]
    best_split_exact(rows,grad,hess)
    assert all((x==y or (math.isnan(x) and math.isnan(y)))
               for a,b in zip(rows,copied) for x,y in zip(a,b))
