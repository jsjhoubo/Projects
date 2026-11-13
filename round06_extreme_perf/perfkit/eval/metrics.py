"""Perplexity, token accuracy and reductions with explicit denominators."""
import math
import torch
from torch.nn import functional as F


def token_nll_sum(logits,targets,ignore_index=-100):
    if logits.shape[:2]!=targets.shape:raise ValueError('shape')
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),
                         ignore_index=ignore_index,reduction='sum')
    return loss


def summarize_nll(total_nll,total_tokens):
    if total_tokens<0:raise ValueError('tokens')
    if total_tokens==0:return {'mean_nll':0.0,'perplexity':1.0}
    mean=float(total_nll)/total_tokens
    return {'mean_nll':mean,'perplexity':math.exp(min(mean,80.0))}


def predictions(logits):return logits.argmax(dim=-1)


def compare_logits(a,b,rtol=1e-4,atol=1e-5):
    if a.shape!=b.shape:raise ValueError('shape mismatch')
    delta=(a.float()-b.float()).abs()
    tolerance=atol+rtol*b.float().abs()
    return {'max_abs':float(delta.max()),
            'mean_abs':float(delta.mean()),
            'fraction_outside':float((delta>tolerance).float().mean())}
