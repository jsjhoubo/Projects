"""Model-memory diagnostics; do not mistake reserved memory for allocated."""
from collections import Counter
import torch


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def trainable_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_bytes(model):
    return sum(p.numel()*p.element_size() for p in model.parameters())


def gradient_bytes(model):
    return sum(p.grad.numel()*p.grad.element_size()
               for p in model.parameters() if p.grad is not None)


def buffer_bytes(model):
    return sum(b.numel()*b.element_size() for b in model.buffers())


def tensor_summary(model):
    dtypes=Counter(str(p.dtype) for p in model.parameters())
    devices=Counter(str(p.device) for p in model.parameters())
    return {'parameters':parameter_count(model),
            'trainable':trainable_parameter_count(model),
            'parameters_bytes':parameter_bytes(model),
            'gradient_bytes':gradient_bytes(model),
            'buffers_bytes':buffer_bytes(model),
            'dtype_counts':dict(dtypes),'device_counts':dict(devices)}


def cuda_snapshot(device):
    dev=torch.device(device)
    if dev.type!='cuda': return None
    return {'allocated':torch.cuda.memory_allocated(dev),
            'reserved':torch.cuda.memory_reserved(dev),
            'peak':torch.cuda.max_memory_allocated(dev)}
