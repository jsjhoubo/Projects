"""Loss reductions are model semantics, not just performance implementation."""
import torch
from torch.nn import functional as F


def causal_token_loss(logits,labels,ignore_index=-100,label_smoothing=0.0):
    if logits.shape[:2]!=labels.shape:raise ValueError('shape')
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),
                           ignore_index=ignore_index,label_smoothing=label_smoothing)


def per_example_losses(logits,labels,ignore_index=-100):
    b,t,v=logits.shape
    losses=F.cross_entropy(logits.reshape(-1,v),labels.reshape(-1),
                           reduction='none',ignore_index=ignore_index).reshape(b,t)
    counts=(labels!=ignore_index).sum(dim=1).clamp_min(1)
    return losses.sum(dim=1)/counts


def masked_accuracy(logits,labels,ignore_index=-100):
    valid=labels!=ignore_index
    denom=valid.sum()
    if int(denom)==0:return 0.0
    predicted=logits.argmax(-1)
    return float(((predicted==labels)&valid).sum().item()/denom.item())


def topk_accuracy(logits,labels,k=5,ignore_index=-100):
    if not 1<=k<=logits.shape[-1]:raise ValueError('k')
    mask=labels!=ignore_index
    denom=int(mask.sum())
    if not denom:return 0.0
    matched=(logits.topk(k,dim=-1).indices==labels.unsqueeze(-1)).any(-1)
    return float((matched&mask).sum().item()/denom)
