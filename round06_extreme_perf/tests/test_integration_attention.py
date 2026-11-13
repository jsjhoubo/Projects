import copy
import pytest
import torch
from perfkit.model.attention import ResearchMultiheadAttention

@pytest.mark.parametrize('heads',[1,2,4])
def test_attention_shape(heads):
    model=ResearchMultiheadAttention(16,heads).eval()
    y=model(torch.randn(2,6,16))
    assert y.shape==(2,6,16)

@pytest.mark.parametrize('length',[1,2,6,11])
def test_attention_cannot_look_ahead(length):
    torch.manual_seed(10)
    model=ResearchMultiheadAttention(16,4).eval()
    x=torch.randn(1,length,16)
    y=x.clone();y[:,-1]=100
    with torch.no_grad():
        a=model(x);b=model(y)
    if length>1:torch.testing.assert_close(a[:,:-1],b[:,:-1],rtol=1e-4,atol=1e-4)

def test_sdpa_close_to_research_when_no_dropout():
    torch.manual_seed(22)
    model=ResearchMultiheadAttention(32,4,dropout=0.).eval()
    x=torch.randn(2,9,32)
    torch.testing.assert_close(model(x,'research'),model(x,'sdpa'),rtol=2e-4,atol=2e-5)

def test_attention_gradients_finite():
    model=ResearchMultiheadAttention(16,4)
    x=torch.randn(2,4,16,requires_grad=True)
    model(x).sum().backward()
    assert torch.isfinite(x.grad).all()
