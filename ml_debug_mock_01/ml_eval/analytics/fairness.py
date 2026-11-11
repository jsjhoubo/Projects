from ._core import *

def group_counts(groups,labels):
    groups,labels=_pairs(groups,labels)
    out=defaultdict(Counter)
    for g,y in zip(groups,labels):out[str(g)][str(y)]+=1
    return {g:dict(c) for g,c in out.items()}

def group_accuracy(pred,truth,groups):
    pred,truth=_pairs(pred,truth)
    if len(groups)!=len(pred):raise ValueError("length mismatch")
    by=defaultdict(lambda:[0,0])
    for a,b,g in zip(pred,truth,groups):
        by[str(g)][0]+=int(a==b)
        by[str(g)][1]+=1
    return {g:correct/count for g,(correct,count) in by.items()}

def min_group_count(groups):
    freq=Counter(groups)
    return min(freq.values(),default=0)

def demographic_parity_gap(predictions,groups):
    predictions,groups=_pairs(predictions,groups)
    totals=Counter(groups)
    positives=Counter(g for p,g in zip(predictions,groups) if p)
    rates=[_safe_div(positives[g],n) for g,n in totals.items()]
    return max(rates)-min(rates) if rates else 0.0

def equal_opportunity_gap(pred,truth,groups):
    pred,truth=_pairs(pred,truth)
    if len(groups)!=len(pred):raise ValueError("length mismatch")
    positives=Counter(g for y,g in zip(truth,groups) if y)
    hits=Counter(g for p,y,g in zip(pred,truth,groups) if p and y)
    rates=[_safe_div(hits[g],n) for g,n in positives.items() if n]
    return max(rates)-min(rates) if rates else 0.0

def missing_group_rate(groups):
    groups=list(groups)
    return _safe_div(sum(g is None for g in groups),len(groups))
