"""Research attention: two implementations with a shared projection state.

The reference backend intentionally materializes full attention scores.
The alternate backend is a comparison target, not necessarily a drop-in for
all masks, dropout settings, backends and dtypes.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .rope import apply_rotary

class ResearchMultiheadAttention(nn.Module):
    def __init__(self,width,heads,dropout=0.0,use_rope=False):
        super().__init__()
        if width%heads:raise ValueError('width/heads')
        self.heads=heads
        self.dim=width//heads
        self.qkv=nn.Linear(width,3*width)
        self.out=nn.Linear(width,width)
        self.dropout=nn.Dropout(dropout)
        self.use_rope=use_rope

    def project(self,x,offset=0):
        batch,seq,width=x.shape
        q,k,v=self.qkv(x).chunk(3,-1)
        def arrange(x):return x.reshape(batch,seq,self.heads,self.dim).transpose(1,2)
        q,k,v=map(arrange,(q,k,v))
        if self.use_rope:q,k=apply_rotary(q,k,offset=offset)
        return q,k,v

    def slow_scores(self,q,k):
        logits=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.dim)
        length=logits.shape[-1]
        block=torch.triu(torch.ones(length,length,device=logits.device,
                                     dtype=torch.bool),diagonal=1)
        return logits.masked_fill(block,float('-inf'))

    def forward(self,x,backend='research',offset=0):
        batch,seq,width=x.shape
        q,k,v=self.project(x,offset)
        if backend=='research':
            outputs=[]
            for head in range(self.heads):
                logit=self.slow_scores(q[:,head:head+1],k[:,head:head+1])
                probs=F.softmax(logit,dim=-1)
                outputs.append(self.dropout(probs)@v[:,head:head+1])
            y=torch.cat(outputs,dim=1)
        elif backend=='sdpa':
            y=F.scaled_dot_product_attention(q,k,v,dropout_p=
                 self.dropout.p if self.training else 0.0,is_causal=True)
        else:raise ValueError('unknown backend')
        return self.out(y.transpose(1,2).contiguous().reshape(batch,seq,width))
