from pathlib import Path
import pytest
import torch
from torch import nn
from perfkit.train.checkpoints import save_checkpoint,load_checkpoint,tensor_sums,build_payload

def test_save_restore_model(tmp_path):
    model=nn.Linear(5,3)
    original={name:t.detach().clone() for name,t in model.state_dict().items()}
    path=tmp_path/'model.pt'
    size=save_checkpoint(path,model,step=7,metadata={'epoch':3})
    assert size>0
    with torch.no_grad():model.weight.add_(42)
    step,metadata=load_checkpoint(path,model)
    assert step==7 and metadata['epoch']==3
    for name,t in model.state_dict().items():
        torch.testing.assert_close(t,original[name])

def test_optimizer_state_roundtrip(tmp_path):
    model=nn.Linear(3,2)
    opt=torch.optim.AdamW(model.parameters(),lr=0.01)
    model(torch.randn(4,3)).sum().backward();opt.step()
    path=tmp_path/'state.pt'
    save_checkpoint(path,model,opt,step=11)
    m2=nn.Linear(3,2);o2=torch.optim.AdamW(m2.parameters(),lr=0.01)
    step,_=load_checkpoint(path,m2,o2)
    assert step==11
    assert len(o2.state)>0

def test_payload_has_expected_keys():
    model=nn.Linear(2,2)
    payload=build_payload(model,step=3)
    assert {'model','step','metadata'}<=set(payload)
    assert tensor_sums(model)
