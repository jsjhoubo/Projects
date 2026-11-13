"""Scenario masked_sum: mask_fusion.

A research engineer receives this torch operation in an experimental pipeline.
Investigate kernel launch counts, allocation pressure, synchronization,
contiguity, dtype behavior and gradient correctness. No optimization is supplied.
Measurement is workload-dependent; CPU wins do not imply CUDA wins.
"""
from __future__ import annotations
import torch


def research_op(x,y=None,scale=1.0):
    """Unoptimized research version; preserve semantics in any rewrite."""
    if x.ndim!=2:
        raise ValueError('expected rank-two input')
    if y is not None and x.shape!=y.shape:
        raise ValueError('shape mismatch')
    if not torch.isfinite(x).all():
        raise ValueError('non-finite x')
    if y is not None and not torch.isfinite(y).all():
        raise ValueError('non-finite y')
    if scale<=0:raise ValueError('scale')
    if y is None:y=torch.zeros_like(x)
    return torch.where(y>0,x*scale,torch.zeros_like(x)).sum(-1,keepdim=True)


def repeated_columns(x, y=None, scale=1.0):
    """Debug-only independent row execution, deliberately slow for large batches."""
    if x.ndim!=2: raise ValueError('expected matrix')
    if y is not None and x.shape != y.shape:raise ValueError('shape mismatch')
    if len(x)==0:return research_op(x,y,scale)
    fragments=[]
    for row in range(x.shape[0]):
        a=x[row:row+1]
        b=None if y is None else y[row:row+1]
        fragments.append(research_op(a,b,scale))
    return torch.cat(fragments,dim=0)


def compare_performance(x,y=None,scale=1.0,repeats=4):
    """Capture both end-to-end time and semantic output; never silently assume speedup."""
    import statistics
    import time
    readings=[]
    output=None
    for _ in range(repeats):
        if x.device.type=='cuda':torch.cuda.synchronize(x.device)
        start=time.perf_counter()
        output=research_op(x,y,scale)
        if x.device.type=='cuda':torch.cuda.synchronize(x.device)
        readings.append(time.perf_counter()-start)
    return {'median_seconds':statistics.median(readings),
            'min_seconds':min(readings),
            'max_seconds':max(readings),
            'shape':tuple(output.shape),
            'input_dtype':str(x.dtype),
            'device':str(x.device),
            'total_elements':x.numel()}


def runtime_stress(n=256,d=128,device='cpu',seed=91):
    """Deterministic workload generation with gradient-sensitive tensors."""
    torch.manual_seed(seed)
    x=torch.randn(n,d,device=device,requires_grad=True)
    y=torch.randn(n,d,device=device)
    return x,y


def gradient_sample(x,y=None,scale=1.0):
    """If output is differentiable, measure aggregate input gradient."""
    x=x.detach().clone().requires_grad_(True)
    result=research_op(x,y,scale)
    if result.numel()==0:return None
    result.sum().backward()
    return x.grad


def repeated_profile(batch_sizes=(8,32,128),dimensions=(32,128,512),device='cpu'):
    """Return a two-dimensional batch/feature scaling table."""
    summary=[]
    for batch in batch_sizes:
        for width in dimensions:
            x,y=runtime_stress(batch,width,device)
            result=compare_performance(x,y,repeats=3)
            result['batch']=batch
            result['width']=width
            summary.append(result)
    return summary
