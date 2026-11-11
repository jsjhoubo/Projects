from ._core import *

def pairwise_agreement(first,second):
    first,second=_pairs(first,second)
    return _safe_div(sum(a==b for a,b in zip(first,second)),len(first))

def cohens_kappa(first,second):
    first,second=_pairs(first,second)
    if not first:return 0.0
    pa=pairwise_agreement(first,second)
    freq_a=Counter(first);freq_b=Counter(second)
    pe=sum(freq_a[key]*freq_b[key] for key in (freq_a.keys()|freq_b.keys()))/(len(first)**2)
    return _safe_div(pa-pe,1-pe)

def majority_vote(votes):
    votes=list(votes)
    if not votes:return None
    counter=Counter(votes)
    return sorted(counter,key=lambda x:(-counter[x],str(x)))[0]

def entropy_of_votes(votes):
    votes=list(votes)
    if not votes:return 0.0
    counter=Counter(votes)
    return -sum((count/len(votes))*math.log(count/len(votes)) for count in counter.values())

def disagreements(first,second):
    first,second=_pairs(first,second)
    return [i for i,(a,b) in enumerate(zip(first,second)) if a!=b]
