"""Batch assembler; a realistic locus for CPU-to-device transfer costs."""
from dataclasses import dataclass
import torch

@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor
    ids: tuple[int, ...]

    def to(self,device,non_blocking=False):
        return TokenBatch(self.input_ids.to(device,non_blocking=non_blocking),
                          self.targets.to(device,non_blocking=non_blocking),
                          self.attention_mask.to(device,non_blocking=non_blocking),
                          self.lengths.to(device,non_blocking=non_blocking),
                          self.ids)

    @property
    def useful_tokens(self): return int(self.attention_mask.sum().item())
    @property
    def padded_tokens(self): return int(self.input_ids.numel())


def pad_batch(records, *, pad=0, device='cpu', max_length=None):
    if not records: raise ValueError('empty batch')
    lengths=[len(e.tokens)-1 for e in records]
    if any(l<1 for l in lengths): raise ValueError('short sequence')
    width=min(max(lengths),max_length or max(lengths))
    xs=torch.full((len(records),width),pad,dtype=torch.long)
    ys=torch.full((len(records),width),-100,dtype=torch.long)
    mask=torch.zeros((len(records),width),dtype=torch.bool)
    for idx,example in enumerate(records):
        length=min(lengths[idx],width)
        xs[idx,:length]=torch.tensor(example.tokens[:length])
        ys[idx,:length]=torch.tensor(example.tokens[1:length+1])
        mask[idx,:length]=True
    return TokenBatch(xs.to(device),ys.to(device),mask.to(device),
                      torch.tensor(lengths).to(device),
                      tuple(e.sample_id for e in records))


def unpad_targets(batch):
    return [batch.targets[i,:int(batch.attention_mask[i].sum())].tolist()
            for i in range(batch.input_ids.shape[0])]


def make_microbatches(records, micro_batch, **kwargs):
    if micro_batch<1:raise ValueError('micro_batch')
    for start in range(0,len(records),micro_batch):
        yield pad_batch(records[start:start+micro_batch],**kwargs)
