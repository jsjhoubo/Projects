"""Rotary positional embeddings with explicit head dimension and offsets."""
import torch


def frequencies(seq,dim,device,dtype,base=10000.0,offset=0):
    if dim%2:raise ValueError('rotary dimension must be even')
    pos=torch.arange(offset,offset+seq,device=device,dtype=torch.float32)
    inv=base**(-torch.arange(0,dim,2,device=device,dtype=torch.float32)/dim)
    angles=pos[:,None]*inv[None,:]
    return torch.cos(angles).to(dtype),torch.sin(angles).to(dtype)


def rotate_half(x):
    a,b=x[...,0::2],x[...,1::2]
    return torch.stack((-b,a),dim=-1).flatten(-2)


def apply_rotary(q,k,offset=0,base=10000.0):
    if q.shape[-1]!=k.shape[-1] or q.ndim!=4 or k.ndim!=4:
        raise ValueError('expect q,k of [batch,heads,sequence,dim]')
    dim=q.shape[-1]
    if dim%2: raise ValueError('head dim must be even')
    c,s=frequencies(q.shape[-2],dim,q.device,q.dtype,base,offset)
    def expand(t):return torch.repeat_interleave(t,2,dim=-1)[None,None,:,:]
    co,si=expand(c),expand(s)
    return q*co+rotate_half(q)*si,k*co+rotate_half(k)*si
