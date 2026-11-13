"""torch.profiler integration; optional trace export for Chrome timeline.

Profiling overhead is nonzero. Do not use measured profiler timing as the
independent speedup benchmark, especially with many tiny CPU operations.
"""
import argparse
import torch
from torch.profiler import profile,ProfilerActivity,record_function
from perfkit.infra.config import ModelConfig
from perfkit.model.decoder import ResearchDecoder
from perfkit.data.corpus import synthetic_corpus
from perfkit.train.optimizers import make_optimizer
from perfkit.train.engine import ResearchTrainer


def profile_experiment(device='cpu',steps=3,trace=None):
    cfg=ModelConfig(vocab_size=128,hidden_size=32,num_heads=4,
                    num_layers=1,intermediate_size=64,max_position_embeddings=80)
    model=ResearchDecoder(cfg).to(device)
    records=synthetic_corpus(32,seed=16,vocab=128,min_length=24,max_length=32)
    trainer=ResearchTrainer(model,make_optimizer(model),device)
    activities=[ProfilerActivity.CPU]
    if torch.device(device).type=='cuda':activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities,record_shapes=True,
                 profile_memory=True,with_stack=False) as prof:
        for _ in range(steps):
            with record_function('research_train_step'):
                trainer.one_step(records[:4],microbatch=2)
            prof.step()
    print(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25))
    if trace:prof.export_chrome_trace(trace)
    return prof


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--trace',default=None)
    a=p.parse_args()
    profile_experiment(a.device,a.steps,a.trace)

if __name__=='__main__':main()
