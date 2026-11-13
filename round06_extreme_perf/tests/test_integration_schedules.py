import pytest
import torch
from perfkit.train.schedules import cosine_lr,linear_lr,Scheduler

@pytest.mark.parametrize('step',[0,1,5,10,99,100])
def test_cosine_lrs_within_bounds(step):
    value=cosine_lr(step,100,10,1e-3,1e-5)
    assert 0<=value<=1e-3

def test_cosine_final_lr():
    assert cosine_lr(100,100,10,1.0,0.1)==pytest.approx(0.1)

def test_linear_decay_monotonic():
    values=[linear_lr(i,10,1.0) for i in range(11)]
    assert all(a>=b for a,b in zip(values,values[1:]))

def test_scheduler_updates_optimizer_group():
    p=torch.nn.Parameter(torch.tensor([1.]))
    opt=torch.optim.SGD([p],lr=1e-2)
    sch=Scheduler(opt,total=10,warmup=2,peak=1e-2)
    a=sch.step();b=sch.step()
    assert a<b
    assert opt.param_groups[0]['lr']==b

def test_invalid_schedule_rejected():
    with pytest.raises(ValueError):cosine_lr(1,5,5,1.0)
