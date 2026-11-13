import torch
from torch import nn
from perfkit.train.optimizers import (
    make_optimizer,global_grad_norm,inspect_gradients,clip_gradients,params_equal)

def test_optimizer_updates():
    model=nn.Linear(2,1)
    opt=make_optimizer(model,lr=0.01)
    before=model.weight.detach().clone()
    model(torch.ones(2,2)).sum().backward()
    opt.step()
    assert not torch.equal(before,model.weight)

def test_zero_gradient_norm():
    model=nn.Linear(2,1)
    assert global_grad_norm(model)==0

def test_norm_reduction():
    model=nn.Linear(3,1)
    model(torch.randn(5,3)).sum().backward()
    norm=global_grad_norm(model)
    assert norm>0
    clip_gradients(model,0.1)
    assert float(global_grad_norm(model))<=0.10001

def test_gradient_diagnostic_fields():
    model=nn.Linear(2,1)
    model(torch.ones(2,2)).sum().backward()
    report=inspect_gradients(model)
    assert report['weight']['finite']
    assert report['bias']['max_abs']>=0

def test_matching_params():
    a=nn.Linear(2,3);b=nn.Linear(2,3)
    b.load_state_dict(a.state_dict())
    assert params_equal(a,b)
