"""Sampling helpers; output distribution is an API contract."""
import torch
from torch.nn import functional as F


def top_k_logits(logits,k):
    if k<0:raise ValueError('k')
    if k==0 or k>=logits.shape[-1]:return logits
    cutoff=torch.topk(logits,k,dim=-1).values[...,-1,None]
    return logits.masked_fill(logits<cutoff,float('-inf'))


def sample_next(logits,temperature=1.0,top_k=0,deterministic=False,
                generator=None):
    if temperature<=0:raise ValueError('temperature')
    scores=top_k_logits(logits.float()/temperature,top_k)
    if deterministic:return scores.argmax(-1,keepdim=True)
    probs=F.softmax(scores,dim=-1)
    return torch.multinomial(probs,1,generator=generator)


def entropy(logits):
    probs=F.softmax(logits,dim=-1)
    return -(probs*F.log_softmax(logits,dim=-1)).sum(-1)
