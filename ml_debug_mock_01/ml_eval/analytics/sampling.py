from ._core import *

def stable_partition(items,ratio,seed=17):
    if not 0<=ratio<=1:raise ValueError("invalid ratio")
    import random
    rng=random.Random(seed)
    values=list(items)
    rng.shuffle(values)
    pivot=int(len(values)*ratio)
    return values[:pivot],values[pivot:]

def stratified_partition(items,key,ratio,seed=17):
    if not 0<=ratio<=1:raise ValueError("invalid ratio")
    groups=defaultdict(list)
    for item in items:groups[key(item)].append(item)
    left=[];right=[]
    for group in sorted(groups,key=str):
        a,b=stable_partition(groups[group],ratio,seed)
        left.extend(a);right.extend(b)
    return left,right

def reservoir(items,k,seed=17):
    if k<0:raise ValueError("negative k")
    import random
    rng=random.Random(seed)
    result=[]
    for idx,item in enumerate(items):
        if idx<k:result.append(item)
        else:
            j=rng.randint(0,idx)
            if j<k:result[j]=item
    return result

def bootstrap_indices(size,draws=None,seed=17):
    if size<1:raise ValueError("size must be positive")
    if draws is None:draws=size
    if draws<0:raise ValueError("negative draws")
    import random
    rng=random.Random(seed)
    return [rng.randrange(size) for _ in range(draws)]

def grouped_fold(groups,n_folds=5,seed=17):
    if n_folds<2:raise ValueError("n_folds must be >= 2")
    import random
    unique=list(dict.fromkeys(groups))
    random.Random(seed).shuffle(unique)
    fold_of={k:i%n_folds for i,k in enumerate(unique)}
    return [fold_of[g] for g in groups]
