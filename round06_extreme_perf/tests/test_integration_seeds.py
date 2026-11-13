import random
import torch
from perfkit.infra.seeds import snapshot_seed,restore_seed,seed_everything,ForkSeed

def test_seed_repeatability_cpu():
    seed_everything(117)
    x=torch.rand(8); y=[random.random() for _ in range(8)]
    seed_everything(117)
    torch.testing.assert_close(torch.rand(8),x)
    assert [random.random() for _ in range(8)]==y

def test_snapshot_and_restore():
    seed_everything(11)
    snap=snapshot_seed()
    expected=torch.rand(5); p=random.random()
    torch.rand(99);random.random()
    restore_seed(snap)
    torch.testing.assert_close(torch.rand(5),expected)
    assert random.random()==p

def test_fork_seed_does_not_change_outer_stream():
    seed_everything(71)
    reference=torch.rand(3); continuation=torch.rand(3)
    seed_everything(71)
    torch.testing.assert_close(torch.rand(3),reference)
    with ForkSeed(31):
        a=torch.rand(10)
        with ForkSeed(31):torch.testing.assert_close(torch.rand(10),a)
    torch.testing.assert_close(torch.rand(3),continuation)

def test_python_rng_fork():
    random.seed(19)
    first=random.random();second=random.random()
    random.seed(19)
    assert random.random()==first
    with ForkSeed(99):[random.random() for _ in range(19)]
    assert random.random()==second
