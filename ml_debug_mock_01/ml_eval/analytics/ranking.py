from ._core import *

def _check_rank(inputs,labels):
    scores,labels=_pairs(inputs,labels)
    scores=_finite(scores)
    labels=_finite(labels)
    return scores,labels

def ranked_indices(scores):
    scores=_finite(scores)
    return sorted(range(len(scores)),key=lambda i:(-scores[i],i))

def dcg(scores,labels,k=None):
    scores,labels=_check_rank(scores,labels)
    order=ranked_indices(scores)
    if k is not None:
        if k<0: raise ValueError("k >= 0 required")
        order=order[:k]
    return sum((2**labels[i]-1)/math.log2(j+2) for j,i in enumerate(order))

def ndcg(scores,labels,k=None):
    scores,labels=_check_rank(scores,labels)
    ideal=dcg(labels,labels,k)
    return _safe_div(dcg(scores,labels,k),ideal)

def reciprocal_rank(scores,labels):
    scores,labels=_check_rank(scores,labels)
    for rank,i in enumerate(ranked_indices(scores),1):
        if labels[i]>0:return 1/rank
    return 0.0

def average_precision(scores,labels):
    scores,labels=_check_rank(scores,labels)
    positives=sum(y>0 for y in labels)
    if not positives:return 0.0
    hit=0
    total=0.0
    for rank,i in enumerate(ranked_indices(scores),1):
        if labels[i]>0:
            hit+=1
            total+=hit/rank
    return total/positives

def recall_at_k(scores,labels,k):
    scores,labels=_check_rank(scores,labels)
    if k<0: raise ValueError("k >= 0 required")
    positives=sum(y>0 for y in labels)
    return _safe_div(sum(labels[i]>0 for i in ranked_indices(scores)[:k]),positives)

def hit_at_k(scores,labels,k):
    return float(recall_at_k(scores,labels,k)>0)

def precision_at_k(scores,labels,k):
    scores,labels=_check_rank(scores,labels)
    if k<=0:raise ValueError("k > 0 required")
    return sum(labels[i]>0 for i in ranked_indices(scores)[:k])/k

def pairwise_accuracy(scores,labels):
    scores,labels=_check_rank(scores,labels)
    hits,total=0.0,0
    for i in range(len(scores)):
        for j in range(i):
            if labels[i]==labels[j]:continue
            total+=1
            direction=(scores[i]-scores[j])*(labels[i]-labels[j])
            hits+=1 if direction>0 else (0.5 if direction==0 else 0)
    return _safe_div(hits,total)
