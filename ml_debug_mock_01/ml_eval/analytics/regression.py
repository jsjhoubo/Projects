from ._core import *

def mae(pred,truth):
    pred,truth=_pairs(pred,truth)
    pred=_finite(pred); truth=_finite(truth)
    return _safe_div(sum(abs(a-b) for a,b in zip(pred,truth)),len(pred))

def mse(pred,truth):
    pred,truth=_pairs(pred,truth)
    pred=_finite(pred); truth=_finite(truth)
    return _safe_div(sum((a-b)**2 for a,b in zip(pred,truth)),len(pred))

def rmse(pred,truth):
    return math.sqrt(mse(pred,truth))

def r2(pred,truth):
    pred,truth=_pairs(pred,truth)
    pred=_finite(pred); truth=_finite(truth)
    if not truth:return 0.0
    mu=mean(truth)
    ss_total=sum((y-mu)**2 for y in truth)
    ss_error=sum((y-p)**2 for y,p in zip(truth,pred))
    return 1-_safe_div(ss_error,ss_total) if ss_total else (1.0 if ss_error==0 else 0.0)

def huber(pred,truth,delta=1.0):
    if delta<=0:raise ValueError("positive delta required")
    pred,truth=_pairs(pred,truth)
    pred=_finite(pred);truth=_finite(truth)
    out=0.0
    for a,b in zip(pred,truth):
        d=abs(a-b)
        out+=0.5*d*d if d<=delta else delta*(d-0.5*delta)
    return _safe_div(out,len(pred))

def pinball(pred,truth,quantile=0.5):
    if not 0<quantile<1:raise ValueError("invalid quantile")
    pred,truth=_pairs(pred,truth)
    return _safe_div(sum(max(quantile*(b-a),(quantile-1)*(b-a)) for a,b in zip(pred,truth)),len(pred))

def smape(pred,truth):
    pred,truth=_pairs(pred,truth)
    return _safe_div(sum(2*abs(a-b)/(abs(a)+abs(b)) if a or b else 0 for a,b in zip(pred,truth)),len(pred))
