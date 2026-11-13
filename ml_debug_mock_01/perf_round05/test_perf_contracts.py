"""Correctness oracles for the shared Resolution/OpenAI/Anthropic-style mock.
Performance improvements must preserve outcomes within documented tolerances.
"""
import math
import torch
from perf_round05.research_model import (
    Config,ResearchDecoder,BatchProvider,make_records,token_loss,
    train_one_step,validate,greedy_decode
)


def cfg():
    return Config(vocab=40,width=16,heads=4,max_seq=20,dropout=0.0)


def test_causal_attention_cannot_see_future():
    torch.manual_seed(1)
    model=ResearchDecoder(cfg()).eval()
    x=torch.tensor([[1,2,3,4,5]])
    y=torch.tensor([[1,2,3,39,38]])
    with torch.no_grad():
        a,b=model(x),model(y)
    torch.testing.assert_close(a[:,:3],b[:,:3],rtol=1e-5,atol=1e-5)


def test_shape_and_dtype():
    model=ResearchDecoder(cfg())
    out=model(torch.randint(0,40,(3,8)))
    assert out.shape==(3,8,40)
    assert torch.isfinite(out).all()


def test_batch_provider_shift_and_order():
    provider=BatchProvider([[1,2,3,4],[5,6,7,8]],"cpu")
    x,y=provider.batch([1,0])
    assert x.tolist()==[[5,6,7],[1,2,3]]
    assert y.tolist()==[[6,7,8],[2,3,4]]


def test_gradients_finite():
    torch.manual_seed(2)
    model=ResearchDecoder(cfg())
    x=torch.randint(1,40,(2,6))
    token_loss(model(x),torch.randint(1,40,(2,6))).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_accumulation_matches_full_batch_gradient():
    torch.manual_seed(3)
    a,b=ResearchDecoder(cfg()),ResearchDecoder(cfg())
    b.load_state_dict(a.state_dict())
    provider=BatchProvider(make_records(8,8,40),"cpu")
    x,y=provider.batch([0,1,2,3])
    token_loss(a(x),y).backward()
    for ids in ([0,1],[2,3]):
        xx,yy=provider.batch(ids)
        (token_loss(b(xx),yy)/2).backward()
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=2e-5)


def test_training_updates_and_validates():
    torch.manual_seed(4)
    model=ResearchDecoder(cfg())
    before=model.embed.weight.detach().clone()
    provider=BatchProvider(make_records(8,8,40),"cpu")
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    result=train_one_step(model,optimizer,provider,[0,1,2,3],
                          microbatch=2,diagnostics=False)
    assert math.isfinite(result)
    assert not torch.equal(before,model.embed.weight)
    assert math.isfinite(validate(model,provider,[4,5]))


def test_greedy_generation_contract():
    torch.manual_seed(5)
    model=ResearchDecoder(cfg())
    out=greedy_decode(model,torch.tensor([[1,2,3],[4,5,6]]),new_tokens=4)
    assert out.shape==(2,7)
    assert bool(((out>=0)&(out<40)).all())
