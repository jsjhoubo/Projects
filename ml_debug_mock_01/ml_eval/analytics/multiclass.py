from ._core import *

def confusion_matrix(pred,truth,labels=None):
    pred,truth=_pairs(pred,truth)
    if labels is None:labels=sorted(set(pred)|set(truth),key=str)
    labels=list(labels)
    keys={name:i for i,name in enumerate(labels)}
    matrix=[[0]*len(labels) for _ in labels]
    for a,b in zip(pred,truth):
        if a not in keys or b not in keys:raise ValueError("unknown class")
        matrix[keys[b]][keys[a]]+=1
    return labels,matrix

def per_class_recall(pred,truth,labels=None):
    labels,matrix=confusion_matrix(pred,truth,labels)
    return {label:_safe_div(matrix[i][i],sum(matrix[i])) for i,label in enumerate(labels)}

def per_class_precision(pred,truth,labels=None):
    labels,matrix=confusion_matrix(pred,truth,labels)
    return {label:_safe_div(matrix[i][i],sum(matrix[j][i] for j in range(len(labels)))) for i,label in enumerate(labels)}

def macro_f1(pred,truth,labels=None):
    p=per_class_precision(pred,truth,labels)
    r=per_class_recall(pred,truth,labels)
    return _safe_div(sum(_safe_div(2*p[k]*r[k],p[k]+r[k]) for k in p),len(p))

def topk_accuracy(logits,labels,k=1):
    logits,labels=_pairs(logits,labels)
    if k<1:raise ValueError("invalid k")
    if not logits:return 0.0
    hits=0
    for row,y in zip(logits,labels):
        ranked=sorted(range(len(row)),key=lambda i:(-row[i],i))
        hits+=int(y in ranked[:k])
    return hits/len(logits)
