"""Randomness bookkeeping for reproducible profiling and regression runs."""
from dataclasses import dataclass
import random
import torch

@dataclass
class SeedSnapshot:
    python_state: object
    torch_cpu: torch.Tensor
    cuda_states: list | None


def snapshot_seed():
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return SeedSnapshot(random.getstate(), torch.get_rng_state().clone(), cuda_states)


def restore_seed(snapshot):
    random.setstate(snapshot.python_state)
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.cuda_states)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class ForkSeed:
    def __init__(self, seed): self.seed=seed; self.snapshot=None
    def __enter__(self):
        self.snapshot=snapshot_seed()
        seed_everything(self.seed)
        return self
    def __exit__(self, typ, exc, tb):
        restore_seed(self.snapshot)
        return False
