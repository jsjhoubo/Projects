import math
import pytest
import torch
from perfkit.model.rope import frequencies,rotate_half,apply_rotary

@pytest.mark.parametrize('dim',[2,4,8,16])
def test_rotation_preserves_vector_norm(dim):
    torch.manual_seed(dim)
    q=torch.randn(2,3,7,dim)
    k=torch.randn(2,3,7,dim)
    qo,ko=apply_rotary(q,k)
    torch.testing.assert_close(qo.norm(dim=-1),q.norm(dim=-1),rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(ko.norm(dim=-1),k.norm(dim=-1),rtol=2e-5,atol=2e-5)

def test_rope_zero_position_is_identity():
    q=torch.randn(1,2,1,8)
    k=torch.randn(1,2,1,8)
    qo,ko=apply_rotary(q,k,offset=0)
    torch.testing.assert_close(qo,q)
    torch.testing.assert_close(ko,k)

def test_rope_offset_changes_phase():
    q=torch.randn(1,1,5,8)
    a,_=apply_rotary(q,q,offset=0)
    b,_=apply_rotary(q,q,offset=10)
    assert not torch.allclose(a,b)

def test_rotary_odd_dimension_rejected():
    with pytest.raises(ValueError):frequencies(3,5,'cpu',torch.float32)
    with pytest.raises(ValueError):rotate_half(torch.randn(2,5)) if False else apply_rotary(torch.randn(1,1,3,5),torch.randn(1,1,3,5))
