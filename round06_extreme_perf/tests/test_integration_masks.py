import pytest
import torch
from perfkit.model.masks import (
    causal_allow_mask,additive_causal_mask,combine_padding_and_causal,
    lengths_to_valid,all_query_rows_have_key)

@pytest.mark.parametrize('n',[1,2,7,19])
def test_causal_shape_and_structure(n):
    mask=causal_allow_mask(n)
    assert mask.shape==(n,n)
    assert mask.dtype==torch.bool
    for i in range(n):
        for j in range(n):assert bool(mask[i,j])==(j<=i)

@pytest.mark.parametrize('n',[1,3,8])
def test_additive_mask_blocks_future(n):
    mask=additive_causal_mask(n)
    assert torch.isfinite(mask.tril()).all()
    for i in range(n):
        for j in range(i+1,n):assert mask[i,j]==float('-inf')

def test_lengths_to_valid_handles_zero_length():
    mask=lengths_to_valid([0,1,3],4)
    assert mask.tolist()==[[False]*4,[True,False,False,False],
                           [True,True,True,False]]

def test_combined_padding_preserves_causality():
    padding=torch.tensor([[1,1,0,0],[1,1,1,0]],dtype=torch.bool)
    mask=combine_padding_and_causal(padding)
    assert mask.shape==(2,4,4)
    assert not bool(mask[0,0,1])
    assert bool(mask[1,2,2])
    assert not bool(mask[1,3,3])
