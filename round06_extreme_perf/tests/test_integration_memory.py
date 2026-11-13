import torch
from torch import nn
from perfkit.infra.memory import (
    parameter_count,trainable_parameter_count,gradient_bytes,parameter_bytes,
    buffer_bytes,tensor_summary,cuda_snapshot)

def test_parameter_count_for_linear_stack():
    model=nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,2))
    assert parameter_count(model)==(4*3+3)+(3*2+2)
    assert trainable_parameter_count(model)==parameter_count(model)

def test_frozen_params_excluded_from_trainable():
    model=nn.Linear(5,3)
    model.bias.requires_grad_(False)
    assert parameter_count(model)==18
    assert trainable_parameter_count(model)==15

def test_gradient_bytes_only_when_present():
    model=nn.Linear(2,1)
    assert gradient_bytes(model)==0
    model(torch.ones(4,2)).sum().backward()
    assert gradient_bytes(model)==parameter_bytes(model)

def test_buffers_are_separate():
    model=nn.BatchNorm1d(5)
    assert buffer_bytes(model)>0
    report=tensor_summary(model)
    assert report['trainable']>0
    assert report['buffers_bytes']==buffer_bytes(model)

def test_cpu_cuda_snapshot_is_none():
    assert cuda_snapshot('cpu') is None
