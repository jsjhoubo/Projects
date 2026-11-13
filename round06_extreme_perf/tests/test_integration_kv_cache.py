import pytest
import torch
from perfkit.model.kv_cache import KVCache

def test_cache_append_and_lengths():
    c=KVCache(layers=2,max_tokens=5)
    k=torch.randn(1,2,3,4)
    c.append(0,k,k,torch.arange(3))
    assert c.snapshot_lengths()==[3,0]
    assert c.memory_bytes()>0

def test_cache_eviction_preserves_last_positions():
    c=KVCache(layers=1,max_tokens=4)
    c.append(0,torch.ones(1,2,3,4),torch.ones(1,2,3,4),torch.arange(3))
    c.append(0,torch.ones(1,2,3,4),torch.ones(1,2,3,4),torch.arange(3,6))
    assert c.entries[0].positions.tolist()==[2,3,4,5]
    assert c.snapshot_lengths()==[4]

def test_cache_clear_releases_entries():
    c=KVCache(layers=3,max_tokens=10)
    c.append(1,torch.randn(1,1,2,8),torch.randn(1,1,2,8),torch.arange(2))
    c.clear()
    assert c.memory_bytes()==0
    assert c.snapshot_lengths()==[0,0,0]

@pytest.mark.parametrize('layer',[-1,3])
def test_bad_layer_rejected(layer):
    c=KVCache(3,10)
    with pytest.raises(IndexError):c.append(layer,torch.ones(1,1,1,4),torch.ones(1,1,1,4),torch.arange(1))
