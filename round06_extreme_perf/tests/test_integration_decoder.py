import pytest
import torch
from perfkit.model.decoder import ResearchDecoder
from perfkit.infra.config import ModelConfig

def tiny(positional='learned'):
    return ModelConfig(vocab_size=44,hidden_size=16,num_heads=4,num_layers=1,
                       intermediate_size=32,max_position_embeddings=24,positional=positional)

@pytest.mark.parametrize('batch,length',[(1,1),(1,6),(2,8)])
def test_model_shapes(batch,length):
    model=ResearchDecoder(tiny())
    ids=torch.randint(0,44,(batch,length))
    assert model(ids).shape==(batch,length,44)

def test_model_tied_weights():
    m=ResearchDecoder(tiny())
    assert m.lm_head.weight is m.embed.weight

def test_model_untied_weights():
    c=ModelConfig(vocab_size=44,hidden_size=16,num_heads=4,num_layers=1,
                  intermediate_size=32,tie_embeddings=False)
    m=ResearchDecoder(c)
    assert m.lm_head.weight is not m.embed.weight

def test_gradients_finite():
    model=ResearchDecoder(tiny())
    output=model(torch.randint(0,44,(2,8))).sum()
    output.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())

def test_oversize_rejected():
    model=ResearchDecoder(tiny())
    with pytest.raises(ValueError):model(torch.randint(0,44,(1,25)))
