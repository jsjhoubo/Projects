"""Autoregressive decode calls the full transformer at every new token."""
import time
import torch
from .sampling import sample_next

@torch.no_grad()
def generate(model,prefix,new_tokens,*,temperature=1.0,top_k=0,
             deterministic=True,backend='research'):
    if prefix.ndim!=2 or prefix.shape[1]<1:raise ValueError('prefix')
    tokens=prefix
    for _ in range(new_tokens):
        context=tokens[:,-model.config.max_position_embeddings:]
        logits=model(context,backend=backend)
        token=sample_next(logits[:,-1],temperature,top_k,deterministic)
        tokens=torch.cat((tokens,token),dim=1)
    return tokens


def benchmark_decode(model,prefix,new_tokens,*,backend='research',repeats=3):
    times=[];out=None
    device=prefix.device
    for _ in range(repeats):
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        out=generate(model,prefix,new_tokens,backend=backend)
        if device.type=='cuda':torch.cuda.synchronize(device)
        times.append(time.perf_counter()-start)
    return {'tokens_generated':prefix.shape[0]*new_tokens,
            'times':times,'last_token':out[:,-1]}
