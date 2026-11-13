"""End-to-end causal decoder and configurable embedding semantics."""
import torch
from torch import nn
from .blocks import TransformerBlock
from .norms import SlowRMSNorm

class ResearchDecoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config.validate()
        self.embed=nn.Embedding(config.vocab_size,config.hidden_size)
        self.positions=(nn.Embedding(config.max_position_embeddings,
                       config.hidden_size) if config.positional=='learned' else None)
        self.blocks=nn.ModuleList(TransformerBlock(config)
                                  for _ in range(config.num_layers))
        self.norm=SlowRMSNorm(config.hidden_size,config.rms_norm_eps)
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        if config.tie_embeddings:self.lm_head.weight=self.embed.weight

    def encode(self,ids,backend='research',positions_offset=0):
        if ids.ndim!=2:raise ValueError('ids rank')
        batch,length=ids.shape
        if length==0 or length+positions_offset>self.config.max_position_embeddings:
            raise ValueError('sequence too long')
        state=self.embed(ids)
        if self.positions is not None:
            pos=torch.arange(positions_offset,positions_offset+length,
                             device=ids.device)
            state=state+self.positions(pos)[None,:,:]
        for block in self.blocks:state=block(state,backend=backend)
        return self.norm(state)

    def forward(self,ids,backend='research'):
        return self.lm_head(self.encode(ids,backend=backend))

    def param_groups(self,weight_decay):
        decay=[]; no_decay=[]
        for name,p in self.named_parameters():
            if p.ndim>=2:decay.append(p)
            else:no_decay.append(p)
        return [{'params':decay,'weight_decay':weight_decay},
                {'params':no_decay,'weight_decay':0.0}]
