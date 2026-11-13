"""Wall-clock timers: make CUDA synchronization assumptions explicit."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict
import math
import statistics
import time
import torch

@dataclass
class Timings:
    sections: dict = field(default_factory=lambda: defaultdict(list))

    def add(self, name, seconds): self.sections[name].append(float(seconds))

    def summary(self):
        return {name: {'min':min(xs),'max':max(xs),
                       'mean':statistics.mean(xs),
                       'median':statistics.median(xs),
                       'p95':sorted(xs)[max(0,math.ceil(.95*len(xs))-1)]}
                for name,xs in self.sections.items() if xs}


def sync_if_cuda(device):
    dev=torch.device(device)
    if dev.type=='cuda': torch.cuda.synchronize(dev)

@contextmanager
def section(name, collector, device='cpu', synchronize=True):
    if synchronize: sync_if_cuda(device)
    t=time.perf_counter()
    try: yield
    finally:
        if synchronize: sync_if_cuda(device)
        collector.add(name,time.perf_counter()-t)


def peak_allocated(device):
    device=torch.device(device)
    return torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None


def reset_peak(device):
    device=torch.device(device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)


def benchmark(fn,warmup=2,repeats=5,device='cpu'):
    if repeats < 1 or warmup < 0: raise ValueError('invalid iteration count')
    for _ in range(warmup): fn()
    timings=Timings()
    for _ in range(repeats):
        with section('iteration',timings,device): fn()
    return timings.summary()['iteration']
