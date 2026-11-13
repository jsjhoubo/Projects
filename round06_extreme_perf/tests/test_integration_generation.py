import pytest
import torch
from perfkit.serve.sampling import top_k_logits,sample_next,entropy
from perfkit.serve.decoding import generate
from perfkit.model.decoder import ResearchDecoder
from perfkit.infra.config import ModelConfig

def make_model():
    return ResearchDecoder(ModelConfig(vocab_size=40,hidden_size=16,num_heads=4,
           num_layers=1,intermediate_size=32,max_position_embeddings=25))

def test_greedy_generation_shape():
    model=make_model().eval()
    out=generate(model,torch.randint(1,40,(2,4)),new_tokens=3)
    assert out.shape==(2,7)

def test_topk_mask_keeps_requested_number():
    logits=torch.arange(10.).reshape(1,10)
    masked=top_k_logits(logits,3)
    assert torch.isfinite(masked).sum()==3

def test_sample_deterministic_argmax():
    logits=torch.tensor([[0.,2.,1.]])
    assert sample_next(logits,deterministic=True).item()==1

def test_entropy_uniform_is_log_vocab():
    logits=torch.zeros(2,8)
    expected=torch.full((2,),torch.log(torch.tensor(8.)).item())
    torch.testing.assert_close(entropy(logits),expected)

def test_invalid_sampling_temperature():
    with pytest.raises(ValueError):sample_next(torch.randn(2,5),temperature=0)
