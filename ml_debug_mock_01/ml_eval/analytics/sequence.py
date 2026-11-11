from ._core import *

def token_accuracy(predictions,targets,ignore_index=None):
    predictions,targets=_pairs(predictions,targets)
    kept=[(a,b) for a,b in zip(predictions,targets) if ignore_index is None or b!=ignore_index]
    return _safe_div(sum(a==b for a,b in kept),len(kept))

def edit_distance(a,b):
    a=list(a);b=list(b)
    dp=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]+[0]*len(b)
        for j,y in enumerate(b,1):
            nxt[j]=min(dp[j]+1,nxt[j-1]+1,dp[j-1]+int(x!=y))
        dp=nxt
    return dp[-1]

def longest_common_subsequence(a,b):
    a=list(a);b=list(b)
    dp=[0]*(len(b)+1)
    for x in a:
        nxt=[0]*(len(b)+1)
        for j,y in enumerate(b,1):
            nxt[j]=dp[j-1]+1 if x==y else max(dp[j],nxt[j-1])
        dp=nxt
    return dp[-1]

def exact_sequence_match(predictions,targets):
    predictions,targets=_pairs(predictions,targets)
    return _safe_div(sum(list(a)==list(b) for a,b in zip(predictions,targets)),len(predictions))

def ngram_counts(tokens,n):
    if n<1:raise ValueError("n >= 1 required")
    tokens=list(tokens)
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def ngram_precision(candidate,reference,n=1):
    a=ngram_counts(candidate,n)
    b=ngram_counts(reference,n)
    hits=sum(min(count,b.get(key,0)) for key,count in a.items())
    return _safe_div(hits,sum(a.values()))

def distinct_ngram_ratio(tokens,n=2):
    grams=ngram_counts(tokens,n)
    return _safe_div(len(grams),sum(grams.values()))
