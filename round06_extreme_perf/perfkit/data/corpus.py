"""Synthetic traces with variable lengths, token skew, and split keys."""
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Example:
    sample_id: int
    user_id: int
    timestamp: int
    tokens: tuple[int, ...]
    fold: int

    @property
    def length(self): return len(self.tokens)


def synthetic_corpus(n, *, seed=17,vocab=384,min_length=16,max_length=64,
                     users=40, skew=0.0):
    if not 2 <= min_length <= max_length: raise ValueError('invalid lengths')
    if users < 1 or vocab < 8: raise ValueError('invalid users/vocab')
    rng=random.Random(seed)
    result=[]
    for i in range(n):
        length=rng.randint(min_length,max_length)
        if rng.random() < skew: length=max_length
        user=rng.randrange(users)
        tokens=[]
        for _ in range(length):
            tokens.append(1 if rng.random() < skew else rng.randrange(2,vocab))
        result.append(Example(i,user,i*7,tuple(tokens),user%5))
    return result


def split_by_time(examples,cutoff):
    train,val=[],[]
    for e in examples:
        (train if e.timestamp<cutoff else val).append(e)
    return train,val


def split_by_user(examples,holdout_users):
    holdout=set(holdout_users)
    train,val=[],[]
    for e in examples:
        (val if e.user_id in holdout else train).append(e)
    return train,val


def corpus_histogram(examples):
    from collections import Counter
    lengths=Counter(e.length for e in examples)
    tokens=Counter(t for e in examples for t in e.tokens)
    users=Counter(e.user_id for e in examples)
    return {'lengths':lengths,'tokens':tokens,'users':users}


def verify_no_overlap(train,val):
    a={e.sample_id for e in train}; b={e.sample_id for e in val}
    if a & b: raise ValueError('train/validation leakage')
    if len(a)!=len(train) or len(b)!=len(val):
        raise ValueError('duplicate sample id')
    return True
