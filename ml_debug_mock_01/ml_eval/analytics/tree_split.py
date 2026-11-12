"""Numeric CART/GBDT split search. Debug correctness and performance.

Finite x <= threshold routes left; finite x > threshold routes right.
NaN follows missing_left. Non-finite derivatives are invalid.
Split gain is the improvement over the unsplit parent.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SplitResult:
    feature: int
    threshold: float
    gain: float
    missing_left: bool
    left_count: int
    right_count: int


def _score(g, h, reg_lambda):
    return g*g/(h+reg_lambda)


def _gain(gl,hl,gr,hr,gt,ht,reg_lambda,gamma):
    return .5*(_score(gl,hl,reg_lambda)+_score(gr,hr,reg_lambda)
               -_score(gt,ht,reg_lambda))-gamma


def _validate(rows,grad,hess,min_child_samples,min_child_weight,reg_lambda,gamma):
    if len(rows)!=len(grad) or len(rows)!=len(hess):
        raise ValueError("inconsistent row counts")
    if min_child_samples<1 or min_child_weight<0 or reg_lambda<=0 or gamma<0:
        raise ValueError("invalid split hyperparameters")
    if any(not math.isfinite(g) for g in grad):
        raise ValueError("non-finite gradient")
    if any(not math.isfinite(h) or h<0 for h in hess):
        raise ValueError("invalid hessian")
    if any(not (math.isfinite(x) or math.isnan(x)) for row in rows for x in row):
        raise ValueError("non-finite feature other than NaN")


def best_split_exact(rows,grad,hess,*,reg_lambda=1.,gamma=0.,
                     min_child_samples=1,min_child_weight=0.):
    """Exhaustive exact split over numeric columns and both missing directions.

    Threshold is a finite observed value strictly below the next distinct value.
    Both children must meet count/Hessian constraints. Zero or negative gain
    yields None. Equal gains tie-break by feature, threshold, missing_left=True.
    """
    _validate(rows,grad,hess,min_child_samples,min_child_weight,reg_lambda,gamma)
    if not rows:
        return None
    d=len(rows[0])
    if any(len(r)!=d for r in rows):
        raise ValueError("ragged matrix")
    gt,ht=sum(grad),sum(hess)
    best=None
    for f in range(d):
        ordered=sorted(((r[f],g,h) for r,g,h in zip(rows,grad,hess)
                        if not math.isnan(r[f])),key=lambda z:z[0])
        # ordered [which bin, grad?, hession? ]
        missing=[(g,h) for r,g,h in zip(rows,grad,hess) if math.isnan(r[f])]
        # missed: [grad?, hession]
        mg,mh,mc=sum(z[0] for z in missing),sum(z[1] for z in missing),len(missing)
        # 
        gl=hl=0.
        for i in range(len(ordered)-1):
            x,g,h=ordered[i]
            gl+=g
            hl+=h
            if x==ordered[i+1][0]:
                continue
            for missing_left in (True,False):
                lg=gl+(mg if missing_left else 0.)
                lh=hl+(mh if missing_left else 0.)
                lc=i+1+(mc if missing_left else 0)
                rg,rh,rc=gt-lg,ht-lh,len(rows)-lc
                if min(lc,rc)<min_child_samples or min(lh,rh)<min_child_weight:
                    continue
                gain=_gain(lg,lh,rg,rh,gt,ht,reg_lambda,gamma)
                candidate=SplitResult(f,x,gain,missing_left,lc,rc)
                if gain>0 and (best is None or gain>best.gain+1e-12 or
                    (abs(gain-best.gain)<=1e-12 and
                    (f,x,not missing_left)<(best.feature,best.threshold,not best.missing_left))):
                    best=candidate
    return best


def best_split_histogram(rows,grad,hess,*,max_bins=16,reg_lambda=1.,
                         gamma=0.,min_child_samples=1,min_child_weight=0.):
    """Approximate numeric split via equal-width bins, preserving tree semantics.

    Returned threshold must be in ORIGINAL feature units, so split_rows
    reproduces the same children and gain on the original matrix.
    """
    if max_bins<2:
        raise ValueError("max_bins must be at least 2")
    if not rows:
        return None
    columns=[]
    for f in range(len(rows[0])):
        finite=[r[f] for r in rows if not math.isnan(r[f])]
        if not finite or min(finite)==max(finite):
            columns.append([float("nan") for _ in rows])
            continue
        lo,hi=min(finite),max(finite)
        width=(hi-lo)/max_bins
        columns.append([float("nan") if math.isnan(r[f]) else
                        min(max_bins-1,int((r[f]-lo)/width)) for r in rows])
    binned=list(zip(*columns))
    split = best_split_exact(binned,grad,hess,reg_lambda=reg_lambda,
                            gamma=gamma,min_child_samples=min_child_samples,
                            min_child_weight=min_child_weight)
    # recover to original feature value
    split_values = [r[f] for r in rows if not math.isnan(r[split.feature])]
    split_values_lo,split_values_hi=min(split_values),max(split_values)
    split_values_width=(split_values_hi-split_values_lo)/max_bins
    split_threshold =split_values_lo + split_values_width * (split.threshold+1)
    return SplitResult(split.feature, split_threshold, split.gain,
                       split.missing_left, split.left_count, split.right_count);


def split_rows(rows,split):
    """Apply split on the original numeric feature matrix."""
    if split is None:
        return list(range(len(rows))),[]
    left,right=[],[]
    for i,row in enumerate(rows):
        x=row[split.feature]
        is_left=split.missing_left if math.isnan(x) else x<=split.threshold
        (left if is_left else right).append(i)
    return left,right
