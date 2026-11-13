import torch
import pytest
from perfkit.model.blocks import TransformerBlock,FeedForward
from perfkit.infra.config import ModelConfig

@pytest.mark.parametrize('activation',['gelu','relu','silu'])
def test_feedforward_shape_and_gradient(activation):
    cfg=ModelConfig(hidden_size=16,num_heads=4,intermediate_size=48,activation=activation)
    layer=FeedForward(cfg)
    x=torch.randn(2,6,16,requires_grad=True)
    y=layer(x)
    assert y.shape==x.shape
    y.sum().backward()
    assert torch.isfinite(x.grad).all()

def test_block_residual_shape():
    cfg=ModelConfig(hidden_size=16,num_heads=4,intermediate_size=32)
    block=TransformerBlock(cfg)
    x=torch.randn(2,5,16)
    assert block(x).shape==x.shape

def test_block_backend_close():
    cfg=ModelConfig(hidden_size=16,num_heads=4,intermediate_size=32,dropout=0)
    block=TransformerBlock(cfg).eval()
    x=torch.randn(2,7,16)
    torch.testing.assert_close(block(x,'research'),block(x,'sdpa'),rtol=1e-4,atol=1e-4)
