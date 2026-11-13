"""Research-style inefficient PyTorch decoder. Performance mock; no answer key.
The algorithm should remain correct as you optimize it; preserve causal semantics.
"""
from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab: int = 512
    width: int = 96
    heads: int = 4
    max_seq: int = 128
    dropout: float = 0.0
    def __post_init__(self):
        if self.width % self.heads: raise ValueError("width must divide heads")


class ResearchAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.width // cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.reshape(b,t,self.heads,self.head_dim)
        k = k.reshape(b,t,self.heads,self.head_dim)
        v = v.reshape(b,t,self.heads,self.head_dim)
        parts = []
        for h in range(self.heads):
            qh,kh,vh=q[:,:,h],k[:,:,h],v[:,:,h]
            logits=qh@kh.transpose(-1,-2)/math.sqrt(self.head_dim)
            mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),diagonal=1)
            logits=logits.masked_fill(mask, float("-inf"))
            weights=self.dropout(F.softmax(logits,dim=-1))
            parts.append(weights@vh)
        return self.proj(torch.cat(parts,-1))


class ResearchDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.embed=nn.Embedding(cfg.vocab,cfg.width)
        self.pos=nn.Embedding(cfg.max_seq,cfg.width)
        self.norm1=nn.LayerNorm(cfg.width)
        self.attn=ResearchAttention(cfg)
        self.norm2=nn.LayerNorm(cfg.width)
        self.ffn=nn.Sequential(nn.Linear(cfg.width,4*cfg.width),
                               nn.GELU(),nn.Linear(4*cfg.width,cfg.width))
        self.head=nn.Linear(cfg.width,cfg.vocab,bias=False)

    def forward(self, x):
        b,t=x.shape
        if t>self.cfg.max_seq: raise ValueError("sequence length too long")
        h=self.embed(x)+self.pos(torch.arange(t,device=x.device))[None]
        h=h+self.attn(self.norm1(h))
        h=h+self.ffn(self.norm2(h))
        return self.head(h)


def make_records(n=256, length=48, vocab=512, seed=91):
    rng=random.Random(seed)
    return [[rng.randrange(1,vocab) for _ in range(length+1)]
            for _ in range(n)]


class BatchProvider:
    def __init__(self, records, device):
        self.records=list(records)
        self.device=torch.device(device)

    def batch(self, ids):
        xs=[]
        ys=[]
        for i in ids:
            rec=self.records[i]
            xs.append(torch.tensor(rec[:-1],dtype=torch.long).to(self.device))
            ys.append(torch.tensor(rec[1:],dtype=torch.long).to(self.device))
        return torch.stack(xs),torch.stack(ys)


def token_loss(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           target.reshape(-1),ignore_index=0)


def train_one_step(model, optimizer, provider, ids, *, microbatch=4, diagnostics=True):
    if not ids or len(ids)%microbatch:
        raise ValueError("requires nonempty full microbatches")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    mean_loss=0.0
    n_micro=len(ids)//microbatch
    for start in range(0,len(ids),microbatch):
        x,y=provider.batch(ids[start:start+microbatch])
        value=token_loss(model(x),y)/n_micro
        value.backward()
        mean_loss+=value.item()
        if diagnostics:
            for param in model.parameters():
                if param.grad is not None:
                    _=param.grad.detach().abs().mean().item()
    nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    return mean_loss


def validate(model, provider, ids):
    model.eval()
    losses=[]
    for i in ids:
        x,y=provider.batch([i])
        # No inference_mode, and metrics are copied one sample at a time.
        losses.append(token_loss(model(x),y).item())
    return sum(losses)/len(losses) if losses else 0.0


@torch.no_grad()
def greedy_decode(model,prefix,new_tokens=12):
    model.eval()
    result=prefix
    for _ in range(new_tokens):
        context=result[:,-model.cfg.max_seq:]
        logits=model(context)
        next_id=logits[:,-1,:].argmax(dim=-1,keepdim=True)
        result=torch.cat((result,next_id),dim=1)
    return result


def benchmark(device=None, *, seq=48,batch=16,repeats=3,diagnostics=True):
    torch.manual_seed(17)
    dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg=Config(max_seq=max(128,seq))
    model=ResearchDecoder(cfg).to(dev)
    provider=BatchProvider(make_records(256,seq,cfg.vocab),dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    if dev.type=="cuda": torch.cuda.reset_peak_memory_stats(dev)
    timings=[]
    for _ in range(repeats):
        if dev.type=="cuda": torch.cuda.synchronize(dev)
        t0=time.perf_counter()
        loss=train_one_step(model,optimizer,provider,list(range(batch)),
                            microbatch=4,diagnostics=diagnostics)
        if dev.type=="cuda": torch.cuda.synchronize(dev)
        timings.append(time.perf_counter()-t0)
    print({"device":str(dev),"seq":seq,"batch":batch,
           "times_seconds":timings,"last_loss":loss,
           "cuda_peak_bytes":torch.cuda.max_memory_allocated(dev)
           if dev.type=="cuda" else None})


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--seq",type=int,default=48)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--no-diagnostics",action="store_true")
    args=parser.parse_args()
    benchmark(seq=args.seq,batch=args.batch,repeats=args.repeats,
              diagnostics=not args.no_diagnostics)
