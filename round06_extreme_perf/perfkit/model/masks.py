"""Mask conventions: True means allowed in multiplicative paths."""
import torch


def causal_allow_mask(length,device='cpu'):
    if length<1:raise ValueError('length')
    return torch.tril(torch.ones(length,length,dtype=torch.bool,device=device))


def additive_causal_mask(length,device='cpu',dtype=torch.float32):
    mask=torch.zeros(length,length,device=device,dtype=dtype)
    return mask.masked_fill(~causal_allow_mask(length,device),float('-inf'))


def combine_padding_and_causal(valid_tokens):
    if valid_tokens.ndim!=2: raise ValueError('expected [batch,sequence]')
    b,t=valid_tokens.shape
    causal=causal_allow_mask(t,valid_tokens.device)
    return causal[None,:,:] & valid_tokens[:,None,:].bool()


def lengths_to_valid(lengths,width):
    lengths=torch.as_tensor(lengths,dtype=torch.long)
    if (lengths<0).any() or (lengths>width).any():raise ValueError('length')
    return torch.arange(width,device=lengths.device)[None,:]<lengths[:,None]


def all_query_rows_have_key(mask):
    if mask.ndim not in (2,3,4):raise ValueError('mask rank')
    return bool(mask.any(dim=-1).all())
