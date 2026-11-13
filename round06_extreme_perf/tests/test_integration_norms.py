import pytest
import torch
from torch import nn
from perfkit.model.norms import SlowRMSNorm,ManualLayerNorm,norm_stats

@pytest.mark.parametrize('width',[4,16,32])
def test_layer_norm_matches_native(width):
    torch.manual_seed(width)
    m=ManualLayerNorm(width)
    r=nn.LayerNorm(width)
    r.weight.data.copy_(m.weight);r.bias.data.copy_(m.bias)
    x=torch.randn(3,7,width)
    torch.testing.assert_close(m(x),r(x),rtol=2e-5,atol=2e-5)

def test_rms_norm_is_finite():
    m=SlowRMSNorm(8)
    y=m(torch.randn(3,8))
    assert torch.isfinite(y).all()

def test_rms_norm_gradient():
    m=SlowRMSNorm(16)
    x=torch.randn(2,16,requires_grad=True)
    m(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert m.weight.grad is not None

def test_norm_statistics_reference():
    x=torch.tensor([1.,2.,3.,4.])
    report=norm_stats(x)
    assert report['mean']==2.5
    assert report['maximum']==4
    assert report['std']>0
