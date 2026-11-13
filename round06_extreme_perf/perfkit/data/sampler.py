"""Sampler variants expose shape skew, scheduling and order preservation."""
import random


def shuffled_indices(size,seed):
    if size<0: raise ValueError('size')
    values=list(range(size))
    random.Random(seed).shuffle(values)
    return values


def bucket_by_length(records,bucket_width=8,seed=17):
    if bucket_width<1:raise ValueError('width')
    groups={}
    for idx,record in enumerate(records):
        bucket=(len(record.tokens)-1)//bucket_width
        groups.setdefault(bucket,[]).append(idx)
    rng=random.Random(seed)
    for indices in groups.values():rng.shuffle(indices)
    keys=list(groups);rng.shuffle(keys)
    return [idx for key in keys for idx in groups[key]]


def distributed_indices(size,world_size,rank,*,seed=17,drop_last=False):
    if world_size<1 or not 0<=rank<world_size:raise ValueError('world size/rank')
    values=shuffled_indices(size,seed)
    if drop_last: values=values[:size//world_size*world_size]
    elif values:
        pad=(-len(values))%world_size
        values.extend(values[:pad])
    return values[rank::world_size]


def aligned_batch_plan(indices,batch_size):
    if batch_size<1:raise ValueError('batch size')
    for i in range(0,len(indices),batch_size):
        yield indices[i:i+batch_size]


def sampling_efficiency(records,indices,batch_size):
    useful=allocated=0
    for batch_ids in aligned_batch_plan(indices,batch_size):
        lengths=[len(records[idx].tokens)-1 for idx in batch_ids]
        useful+=sum(lengths)
        allocated+=max(lengths)*len(lengths)
    return useful/allocated if allocated else 0.0
