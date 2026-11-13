import math
import pytest
import torch
from perfkit.eval.metrics import token_nll_sum,summarize_nll,compare_logits
from perfkit.eval.evaluator import evaluate_inference_mode
from perfkit.model.decoder import ResearchDecoder
from perfkit.infra.config import ModelConfig
from perfkit.data.corpus import synthetic_corpus


def model():
    return ResearchDecoder(ModelConfig(vocab_size=40,hidden_size=16,
             num_heads=4,num_layers=1,intermediate_size=32,max_position_embeddings=30))

def test_validation_result_finite():
    m=model();data=synthetic_corpus(2,vocab=40,min_length=4,max_length=9)
    result=evaluate_inference_mode(m,data)
    assert math.isfinite(result.mean_nll)
    assert result.perplexity>=1
    assert result.valid_tokens>0
    assert result.examples==2

def test_mean_nll_and_perplexity():
    report=summarize_nll(10.0,5)
    assert report['mean_nll']==2
    assert report['perplexity']==pytest.approx(math.exp(2))

def test_compare_logits_identity():
    x=torch.randn(2,4,8)
    result=compare_logits(x,x)
    assert result['max_abs']==0
    assert result['fraction_outside']==0

def test_nll_sum_matches_criterion():
    x=torch.randn(2,3,8)
    y=torch.randint(0,8,(2,3))
    ref=torch.nn.functional.cross_entropy(x.reshape(-1,8),y.reshape(-1),reduction='sum')
    torch.testing.assert_close(token_nll_sum(x,y),ref)
