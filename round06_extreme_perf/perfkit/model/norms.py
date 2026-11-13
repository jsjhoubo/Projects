"""Normalization kernels and non-fused implementations for profiling."""
import torch
from torch import nn

class SlowRMSNorm(nn.Module):
    def __init__(self,width,eps=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(width))
        self.eps=eps

    def forward(self,x):
        squared=x.float().pow(2)
        variances=squared.mean(dim=-1,keepdim=True)
        inv=torch.rsqrt(variances+self.eps)
        return (x.float()*inv).to(dtype=x.dtype)*self.weight

class ManualLayerNorm(nn.Module):
    def __init__(self,width,eps=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(width))
        self.bias=nn.Parameter(torch.zeros(width))
        self.eps=eps

    def forward(self,x):
        avg=x.mean(dim=-1,keepdim=True)
        centered=x-avg
        variance=centered.pow(2).mean(dim=-1,keepdim=True)
        return centered*torch.rsqrt(variance+self.eps)*self.weight+self.bias

def norm_stats(x):
    return {'mean':x.float().mean().item(),
            'std':x.float().std(unbiased=False).item(),
            'maximum':x.abs().amax().item()}
