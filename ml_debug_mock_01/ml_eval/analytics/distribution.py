from ._core import *

def normalized_histogram(values,bounds):
    values=_finite(values);bounds=_finite(bounds)
    if sorted(bounds)!=bounds or len(set(bounds))!=len(bounds):raise ValueError("invalid bounds")
    counts=[0]*(len(bounds)+1)
    for value in values:
        pos=0
        while pos<len(bounds) and value>=bounds[pos]:
            pos+=1
        counts[pos]+=1
    return [_safe_div(count,len(values)) for count in counts]

def total_variation(p,q):
    p,q=_pairs(p,q)
    return sum(abs(a-b) for a,b in zip(_finite(p),_finite(q)))/2

def kl_divergence(p,q,eps=1e-12):
    p,q=_pairs(p,q)
    if eps<=0:raise ValueError("eps > 0 required")
    return sum(a*math.log(max(a,eps)/max(b,eps)) for a,b in zip(p,q) if a>0)

def js_divergence(p,q):
    p,q=_pairs(p,q)
    m=[(a+b)/2 for a,b in zip(p,q)]
    return (kl_divergence(p,m)+kl_divergence(q,m))/2

def psi(reference,current,eps=1e-8):
    p,q=_pairs(reference,current)
    return sum((a-b)*math.log(max(a,eps)/max(b,eps)) for a,b in zip(p,q))

def categorical_frequency(items):
    result=Counter(str(x) for x in items)
    total=sum(result.values())
    return {key:_safe_div(value,total) for key,value in sorted(result.items())}

def empirical_cdf(values,query):
    values=sorted(_finite(values))
    return [_safe_div(sum(x<=v for x in values),len(values)) for v in query]

def quantile(values,q):
    values=sorted(_finite(values))
    if not values:raise ValueError("empty values")
    if not 0<=q<=1:raise ValueError("quantile out of range")
    h=(len(values)-1)*q
    lo=int(h);hi=min(lo+1,len(values)-1)
    return values[lo]+(h-lo)*(values[hi]-values[lo])
