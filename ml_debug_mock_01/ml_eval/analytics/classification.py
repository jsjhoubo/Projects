from ._core import *

def confusion(pred, truth):
    pred,truth=_pairs(pred,truth)
    pred=_check_binary(pred); truth=_check_binary(truth)
    tp=sum(a==1 and b==1 for a,b in zip(pred,truth))
    fp=sum(a==1 and b==0 for a,b in zip(pred,truth))
    fn=sum(a==0 and b==1 for a,b in zip(pred,truth))
    tn=len(pred)-tp-fp-fn
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn}

def precision(pred,truth):
    c=confusion(pred,truth)
    return _safe_div(c["tp"],c["tp"]+c["fp"])

def recall(pred,truth):
    c=confusion(pred,truth)
    return _safe_div(c["tp"],c["tp"]+c["fn"])

def specificity(pred,truth):
    c=confusion(pred,truth)
    return _safe_div(c["tn"],c["tn"]+c["fp"])

def fbeta(pred,truth,beta=1.0):
    if beta<=0: raise ValueError("beta > 0 required")
    p,r=precision(pred,truth),recall(pred,truth)
    return _safe_div((1+beta**2)*p*r,beta**2*p+r)

def accuracy(pred,truth):
    pred,truth=_pairs(pred,truth)
    return _safe_div(sum(a==b for a,b in zip(pred,truth)),len(pred))

def balanced_accuracy(pred,truth):
    return (recall(pred,truth)+specificity(pred,truth))/2

def matthews(pred,truth):
    c=confusion(pred,truth)
    tp,tn,fp,fn=(c[k] for k in ("tp","tn","fp","fn"))
    denom=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return _safe_div(tp*tn-fp*fn,denom)

def threshold_scores(scores,threshold=0.5):
    return [int(float(s)>=threshold) for s in scores]

def threshold_sweep(scores,labels,thresholds):
    scores,labels=_pairs(scores,labels)
    return [(t,fbeta(threshold_scores(scores,t),labels)) for t in thresholds]

def class_weighted_accuracy(pred,truth,weights):
    pred,truth=_pairs(pred,truth)
    if len(weights)!=len(pred): raise ValueError("length mismatch")
    w=_finite(weights)
    if any(x<0 for x in w): raise ValueError("negative sample weight")
    return _safe_div(sum(weight for a,b,weight in zip(pred,truth,w) if a==b),sum(w))
