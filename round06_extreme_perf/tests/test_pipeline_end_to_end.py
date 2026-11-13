"""End-to-end tests verify cross-module semantics, not microbenchmark timings."""
import math
import torch
import pytest
from perfkit.infra.config import ModelConfig
from perfkit.infra.seeds import seed_everything
from perfkit.model.decoder import ResearchDecoder
from perfkit.data.corpus import synthetic_corpus
from perfkit.data.batcher import pad_batch
from perfkit.train.engine import ResearchTrainer
from perfkit.train.optimizers import make_optimizer
from perfkit.eval.evaluator import evaluate_inference_mode
from perfkit.serve.decoding import generate


def minimal_model():
    cfg=ModelConfig(vocab_size=64,hidden_size=16,num_heads=4,num_layers=1,
                    intermediate_size=32,max_position_embeddings=32)
    return ResearchDecoder(cfg)


def test_one_step_changes_weights_and_produces_valid_metrics():
    seed_everything(15)
    model=minimal_model()
    data=synthetic_corpus(12,vocab=64,min_length=6,max_length=10)
    baseline=evaluate_inference_mode(model,data[:2])
    before=model.embed.weight.detach().clone()
    result=ResearchTrainer(model,make_optimizer(model)).one_step(data[:4],microbatch=2)
    after=evaluate_inference_mode(model,data[:2])
    assert result.loss>0
    assert result.examples==4
    assert result.tokens>0
    assert result.elapsed>=0
    assert not torch.equal(before,model.embed.weight)
    assert math.isfinite(baseline.mean_nll) and math.isfinite(after.mean_nll)


def test_generation_keeps_prefix():
    seed_everything(71)
    model=minimal_model().eval()
    prefix=torch.randint(1,64,(2,5))
    output=generate(model,prefix,3)
    assert output.shape==(2,8)
    torch.testing.assert_close(output[:,:5],prefix)


def test_different_batching_same_padded_tokens():
    data=synthetic_corpus(9,vocab=64,min_length=6,max_length=15)
    a=pad_batch(data[:4])
    b=pad_batch(data[:2]);c=pad_batch(data[2:4])
    assert a.useful_tokens==b.useful_tokens+c.useful_tokens


def test_model_backend_consistency():
    seed_everything(19)
    model=minimal_model().eval()
    ids=torch.randint(0,64,(2,8))
    with torch.no_grad():
        a=model(ids,'research');b=model(ids,'sdpa')
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)


def test_adversarial_seqlen_variance():
    records=synthetic_corpus(30,vocab=64,min_length=3,max_length=30,skew=.4)
    padded=pad_batch(records)
    assert padded.input_ids.shape[0]==30
    assert padded.padded_tokens>=padded.useful_tokens


def test_position_tracking_under_causal_generation():
    model=minimal_model().eval()
    prefix=torch.randint(0,64,(1,20))
    tokens=generate(model,prefix,new_tokens=14)
    assert tokens.shape==(1,34)
    assert int(tokens.max())<64
