"""Transformer layer assembly with intentionally unfused subgraphs."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import ResearchMultiheadAttention
from .norms import SlowRMSNorm

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.up=nn.Linear(config.hidden_size,config.intermediate_size)
        self.down=nn.Linear(config.intermediate_size,config.hidden_size)
        self.activation=config.activation
        self.dropout=nn.Dropout(config.dropout)

    def forward(self,x):
        h=self.up(x)
        if self.activation=='gelu':h=F.gelu(h)
        elif self.activation=='silu':h=F.silu(h)
        else:h=F.relu(h)
        return self.dropout(self.down(h))

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.pre_attn=SlowRMSNorm(config.hidden_size,config.rms_norm_eps)
        self.attn=ResearchMultiheadAttention(config.hidden_size,config.num_heads,
                 config.dropout,config.positional=='rope')
        self.pre_ffn=SlowRMSNorm(config.hidden_size,config.rms_norm_eps)
        self.ffn=FeedForward(config)
        self.drop=nn.Dropout(config.dropout)

    def forward(self,x,backend='research'):
        x=x+self.drop(self.attn(self.pre_attn(x),backend=backend))
        x=x+self.drop(self.ffn(self.pre_ffn(x)))
        return x
