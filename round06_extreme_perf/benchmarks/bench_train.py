"""Representative end-to-end training benchmark with deliberate scale sweeps.

This is a diagnostic, never a flaky pass/fail performance assertion.
Run from repository root: python -m benchmarks.bench_train --device cpu
"""
import argparse
import json
import statistics
import torch
from perfkit.infra.config import ModelConfig
from perfkit.infra.seeds import seed_everything
from perfkit.infra.memory import cuda_snapshot
from perfkit.infra.timing import reset_peak
from perfkit.data.corpus import synthetic_corpus
from perfkit.model.decoder import ResearchDecoder
from perfkit.train.optimizers import make_optimizer
from perfkit.train.engine import ResearchTrainer


def run(device,batch,seq,heads=4,layers=2,width=96,steps=4,
        diagnostics=True,backend='research'):
    seed_everything(2026)
    cfg=ModelConfig(vocab_size=384,hidden_size=width,num_heads=heads,
                    num_layers=layers,intermediate_size=width*4,
                    max_position_embeddings=seq+8).validate()
    model=ResearchDecoder(cfg).to(device)
    records=synthetic_corpus(max(64,batch*2),seed=21,vocab=cfg.vocab_size,
                              min_length=seq+1,max_length=seq+1)
    optimizer=make_optimizer(model)
    trainer=ResearchTrainer(model,optimizer,device,diagnostics,backend)
    # One warm-up step is essential for meaningful CUDA kernels.
    trainer.one_step(records[:batch],microbatch=min(4,batch))
    reset_peak(device)
    measurements=[]
    for step in trainer.train_steps(records,steps,batch,microbatch=min(4,batch)):
        measurements.append(step.elapsed)
    total_tokens=sum(batch*seq for _ in range(steps))
    return {'device':str(device),'backend':backend,'seq':seq,'batch':batch,
            'hidden':width,'layers':layers,'heads':heads,
            'median_step_s':statistics.median(measurements),
            'total_tokens':total_tokens,
            'throughput_tokens_s':total_tokens/sum(measurements),
            'memory':cuda_snapshot(device)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--seq',type=int,default=32)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--layers',type=int,default=1)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--backend',choices=['research','sdpa'],default='research')
    parser.add_argument('--no-diagnostics',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.device,args.batch,args.seq,width=args.width,
           layers=args.layers,steps=args.steps,backend=args.backend,
           diagnostics=not args.no_diagnostics),indent=2))

if __name__=='__main__':main()
