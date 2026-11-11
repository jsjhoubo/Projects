from ._core import *

def bootstrap_mean_interval(values,repeats=200,confidence=0.95,seed=17):
    values=_finite(values)
    if not values:raise ValueError("empty values")
    if repeats<2:raise ValueError("need repeats")
    if not 0<confidence<1:raise ValueError("confidence must be in (0,1)")
    import random
    rng=random.Random(seed)
    stats=[]
    for _ in range(repeats):
        draw=[values[rng.randrange(len(values))] for _ in values]
        stats.append(sum(draw)/len(draw))
    stats.sort()
    lo=int(((1-confidence)/2)*(repeats-1))
    hi=int(((1+confidence)/2)*(repeats-1))
    return stats[lo],stats[hi]

def paired_deltas(before,after):
    before,after=_pairs(before,after)
    return [b-a for a,b in zip(before,after)]

def paired_bootstrap(before,after,repeats=200,seed=17):
    return bootstrap_mean_interval(paired_deltas(before,after),repeats,seed=seed)

def mean_std(values):
    values=_finite(values)
    if not values:return 0.0,0.0
    mu=mean(values)
    return mu,math.sqrt(sum((x-mu)**2 for x in values)/len(values))

def stderr(values):
    values=_finite(values)
    if not values:return 0.0
    _,sd=mean_std(values)
    return sd/math.sqrt(len(values))

def wilson_interval(successes,n,z=1.96):
    if n<0 or not 0<=successes<=n:raise ValueError("invalid count")
    if n==0:return 0.0,1.0
    p=successes/n
    d=1+z*z/n
    center=(p+z*z/(2*n))/d
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.0,center-radius),min(1.0,center+radius)
