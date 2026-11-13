"""Cache container; cache correctness is distinct from attention correctness."""
from dataclasses import dataclass
import torch

@dataclass
class KVEntry:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor

class KVCache:
    def __init__(self,layers,max_tokens):
        if layers<1 or max_tokens<1:raise ValueError('layers/max_tokens')
        self.layers=layers
        self.max_tokens=max_tokens
        self.entries=[None]*layers

    def clear(self):self.entries=[None]*self.layers

    def __len__(self):
        x=self.entries[0]
        return 0 if x is None else x.key.shape[-2]

    def append(self,layer,key,value,positions):
        if not 0<=layer<self.layers:raise IndexError(layer)
        if key.shape!=value.shape or key.ndim!=4:raise ValueError('K/V shape')
        if positions.numel()!=key.shape[-2]:raise ValueError('positions')
        last=self.entries[layer]
        if last is None:
            self.entries[layer]=KVEntry(key,value,positions)
        else:
            joined_k=torch.cat((last.key,key),dim=-2)
            joined_v=torch.cat((last.value,value),dim=-2)
            joined_pos=torch.cat((last.positions,positions),dim=0)
            self.entries[layer]=KVEntry(joined_k[:,:,-self.max_tokens:],
                                         joined_v[:,:,-self.max_tokens:],
                                         joined_pos[-self.max_tokens:])
        return self.entries[layer]

    def memory_bytes(self):
        return sum((x.key.numel()*x.key.element_size()+
                    x.value.numel()*x.value.element_size())
                   for x in self.entries if x is not None)

    def snapshot_lengths(self):
        return [0 if x is None else x.key.shape[-2] for x in self.entries]
