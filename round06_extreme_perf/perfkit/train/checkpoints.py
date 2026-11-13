"""Checkpoint and resume performance: serialization vs GPU work."""
from pathlib import Path
import torch


def build_payload(model,optimizer=None,step=0,metadata=None):
    payload={'model':model.state_dict(),'step':int(step),
             'metadata':dict(metadata or {})}
    if optimizer is not None:payload['optimizer']=optimizer.state_dict()
    return payload


def save_checkpoint(path,model,optimizer=None,step=0,metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=build_payload(model,optimizer,step,metadata)
    torch.save(payload,path)
    return path.stat().st_size


def load_checkpoint(path,model,optimizer=None,map_location='cpu'):
    payload=torch.load(path,map_location=map_location,weights_only=True)
    model.load_state_dict(payload['model'])
    if optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
    return payload['step'],payload.get('metadata',{})


def tensor_sums(model):
    return {name:float(p.detach().float().sum())
            for name,p in model.named_parameters()}
