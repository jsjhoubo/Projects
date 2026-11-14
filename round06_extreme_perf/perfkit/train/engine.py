"""Inefficient but readable experimental loop; optimize end-to-end, not locally.

Its diagnostics are intentionally expensive and can dominate GPU workloads.
Train semantics and reduction denominators remain correctness requirements.
"""
from dataclasses import dataclass
import time
import torch
from torch import nn
from .losses import causal_token_loss
from .optimizers import inspect_gradients,clip_gradients
from ..data.batcher import pad_batch

@dataclass
class TrainStep:
    loss: float
    examples: int
    tokens: int
    elapsed: float
    gradient_summary: dict

class ResearchTrainer:
    def __init__(self,model,optimizer,device='cpu',diagnostics=True,backend='research',diagnostics_interval=3):
        self.model=model
        self.optimizer=optimizer
        self.device=torch.device(device)
        self.diagnostics=diagnostics
        if diagnostics_interval < 1:raise ValueError('diagnostics_interval must be >= 1')
        self.diagnostics_interval=diagnostics_interval
        self.backend=backend
        self.steps=0

    def one_step(self,examples,microbatch=4,clip_norm=1.0):
        if not examples:raise ValueError('empty examples')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        t0=time.perf_counter()
        chunks=[examples[x:x+microbatch] for x in range(0,len(examples),microbatch)]
        total=0.0
        token_count=0
        for chunk in chunks:
            batch=pad_batch(chunk,device=self.device,
                            max_length=self.model.config.max_position_embeddings)
            logits=self.model(batch.input_ids,backend=self.backend)
            loss=causal_token_loss(logits,batch.targets)/len(chunks)
            loss.backward()
            total+=float(loss.item())
            token_count+=batch.useful_tokens
        # Diagnostics run every N training steps; clipping remains every step.
        should_inspect=self.diagnostics and (self.steps+1)%self.diagnostics_interval==0
        stats=inspect_gradients(self.model) if should_inspect else {}
        clip_gradients(self.model,clip_norm)
        self.optimizer.step()
        self.steps+=1
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        return TrainStep(total,len(examples),token_count,
                         time.perf_counter()-t0,stats)

    def train_steps(self,examples,n_steps,batch_size,microbatch=4):
        results=[]
        for i in range(n_steps):
            start=(i*batch_size)%max(1,len(examples))
            batch=examples[start:start+batch_size]
            if not batch:batch=examples[:batch_size]
            results.append(self.one_step(batch,microbatch))
        return results
