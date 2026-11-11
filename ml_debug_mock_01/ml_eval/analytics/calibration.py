from ._core import *

def brier(scores,labels):
    scores,labels=_pairs(scores,labels)
    scores=_finite(scores)
    labels=_check_binary(labels)
    return _safe_div(sum((p-y)**2 for p,y in zip(scores,labels)),len(scores))

def logloss(scores,labels,eps=1e-12):
    scores,labels=_pairs(scores,labels)
    scores=_finite(scores)
    labels=_check_binary(labels)
    if eps<=0 or eps>=0.5:raise ValueError("bad eps")
    return _safe_div(sum(-(y*math.log(max(eps,min(1-eps,p)))+(1-y)*math.log(max(eps,min(1-eps,1-p)))) for p,y in zip(scores,labels)),len(scores))

def reliability_bins(scores,labels,bins=10):
    scores,labels=_pairs(scores,labels)
    scores=_finite(scores); labels=_check_binary(labels)
    if bins<1:raise ValueError("bins must be positive")
    counts=[0]*bins; sum_p=[0.0]*bins; sum_y=[0.0]*bins
    for p,y in zip(scores,labels):
        if p<0 or p>1:raise ValueError("probability out of range")
        ix=min(int(p*bins),bins-1)
        counts[ix]+=1;sum_p[ix]+=p;sum_y[ix]+=y
    return [(counts[i],_safe_div(sum_p[i],counts[i]),_safe_div(sum_y[i],counts[i])) for i in range(bins)]

def ece(scores,labels,bins=10):
    rb=reliability_bins(scores,labels,bins)
    n=sum(count for count,_,_ in rb)
    return _safe_div(sum(count*abs(pred-actual) for count,pred,actual in rb),n)

def mce(scores,labels,bins=10):
    return max((abs(pred-actual) for count,pred,actual in reliability_bins(scores,labels,bins) if count),default=0.0)

def temperature_scale_logits(logits,temperature):
    if temperature<=0:raise ValueError("temperature > 0 required")
    return [float(x)/temperature for x in logits]

def sigmoid(x):
    if x>=0:
        z=math.exp(-x)
        return 1/(1+z)
    z=math.exp(x)
    return z/(1+z)

def softmax(logits):
    logits=_finite(logits)
    if not logits:return []
    mx=max(logits)
    ex=[math.exp(z-mx) for z in logits]
    denom=sum(ex)
    return [e/denom for e in ex]

def entropy(probs):
    probs=_finite(probs)
    if any(x<0 for x in probs):raise ValueError("negative probability")
    return -sum(p*math.log(p) for p in probs if p>0)
