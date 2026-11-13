from collections import Counter
import pytest
from perfkit.data.corpus import synthetic_corpus
from perfkit.data.sampler import (
    shuffled_indices,bucket_by_length,distributed_indices,
    aligned_batch_plan,sampling_efficiency)

@pytest.mark.parametrize('size',[0,1,12,100])
def test_shuffle_is_permutation(size):
    a=shuffled_indices(size,11)
    assert sorted(a)==list(range(size))
    assert a==shuffled_indices(size,11)

def test_bucket_by_length_is_permutation():
    records=synthetic_corpus(111)
    indices=bucket_by_length(records,bucket_width=8)
    assert sorted(indices)==list(range(111))

def test_distributed_drop_last_exact_disjoint():
    shards=[distributed_indices(18,3,r,drop_last=True) for r in range(3)]
    assert all(len(s)==6 for s in shards)
    assert sorted(i for s in shards for i in s)==list(range(18))

@pytest.mark.parametrize('rank',[0,1,2,3])
def test_distributed_no_drop_has_equal_lengths(rank):
    shard=distributed_indices(17,4,rank,drop_last=False)
    assert len(shard)==5

def test_packing_efficiency_in_unit_interval():
    records=synthetic_corpus(44,min_length=5,max_length=100)
    indices=list(range(44))
    efficiency=sampling_efficiency(records,indices,batch_size=8)
    assert 0<efficiency<=1

def test_planned_batch_includes_tail():
    values=list(range(17))
    assert sum(aligned_batch_plan(values,4),[])==values
    with pytest.raises(ValueError):list(aligned_batch_plan(values,0))
