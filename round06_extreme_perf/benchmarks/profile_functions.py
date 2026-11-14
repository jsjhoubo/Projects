"""Opt-in, function-oriented PyTorch profiler for the existing training implementation.

No changes to trainer, model, or benchmark code are required. Wrappers are
temporary, preserving the actual code path (including local optimizations).
Run from the round06_extreme_perf directory:
  python -m benchmarks.profile_functions --device cuda --steps 5
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import functools
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from perfkit.infra.config import ModelConfig
from perfkit.model.decoder import ResearchDecoder
from perfkit.model.attention import ResearchMultiheadAttention
from perfkit.model.blocks import FeedForward, TransformerBlock
from perfkit.model.norms import SlowRMSNorm
from perfkit.data.corpus import synthetic_corpus
from perfkit.train import engine
from perfkit.train.optimizers import make_optimizer


@contextlib.contextmanager
def labeled_functions(model, optimizer):
    """Label *real* function calls; restore all attributes on exit.

    Some functions invoke other labeled functions, so use Self CPU/CUDA columns
    for non-overlapping timing and CPU/CUDA total for inclusive attribution.
    """
    undo=[]

    def patch(obj, name, label):
        original=getattr(obj,name)
        @functools.wraps(original)
        def wrapper(*args,**kwargs):
            with record_function("fn::"+label):
                return original(*args,**kwargs)
        setattr(obj,name,wrapper)
        undo.append((obj,name,original))

    try:
        # Functions referenced directly by engine.py at runtime.
        for name in ("pad_batch","causal_token_loss",
                     "inspect_gradients","clip_gradients"):
            patch(engine,name,"train."+name)
        patch(optimizer,"step","optim.AdamW.step")
        # PyTorch Tensor.backward eventually invokes this Python entry point.
        patch(torch.autograd,"backward","autograd.backward")
        # nn.Module.__call__ dispatches to each instance's forward method.
        patch(model,"forward","model.ResearchDecoder.forward")
        patch(model,"encode","model.ResearchDecoder.encode")
        for ix,block in enumerate(model.blocks):
            patch(block,"forward",f"model.block[{ix}].forward")
            patch(block.attn,"forward",f"model.block[{ix}].attention")
            patch(block.attn,"project",f"model.block[{ix}].qkv_project")
            patch(block.attn,"slow_scores",f"model.block[{ix}].slow_scores")
            patch(block.ffn,"forward",f"model.block[{ix}].ffn")
            patch(block.pre_attn,"forward",f"model.block[{ix}].pre_attn_norm")
            patch(block.pre_ffn,"forward",f"model.block[{ix}].pre_ffn_norm")
        patch(model.norm,"forward","model.final_norm")
        yield
    finally:
        for obj,name,original in reversed(undo):
            setattr(obj,name,original)


def profile_functions(device="cuda",steps=5,backend="research",trace=None,csv_path=None):
    dev=torch.device(device)
    if dev.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg=ModelConfig(vocab_size=128,hidden_size=32,num_heads=4,
                    num_layers=1,intermediate_size=64,
                    max_position_embeddings=80)
    model=ResearchDecoder(cfg).to(dev)
    records=synthetic_corpus(32,seed=16,vocab=128,min_length=24,max_length=32)
    optimizer=make_optimizer(model)
    trainer=engine.ResearchTrainer(model,optimizer,dev,backend=backend)
    activities=[ProfilerActivity.CPU]
    if dev.type=="cuda": activities.append(ProfilerActivity.CUDA)
    with labeled_functions(model,optimizer):
        with profile(activities=activities,record_shapes=True,
                     profile_memory=True,with_stack=False) as p:
            for _ in range(steps):
                with record_function("fn::train.ResearchTrainer.one_step"):
                    trainer.one_step(records[:4],microbatch=2)
                p.step()
    events=[ev for ev in p.key_averages() if ev.key.startswith("fn::")]
    events.sort(key=lambda ev:ev.self_cpu_time_total,reverse=True)
    print("\nFunction-level attribution (times are aggregated over all steps)")
    print(f'{"Function":48} {"Calls":>7} {"Self CPU ms":>13} '
          f'{"CPU total ms":>13} {"Self CUDA ms":>14} {"CUDA total ms":>14}')
    for ev in events:
        print(f"{ev.key[4:48]:48} {ev.count:7d} "
              f"{ev.self_cpu_time_total/1000:13.3f} "
              f"{ev.cpu_time_total/1000:13.3f} "
              f"{ev.self_device_time_total/1000:14.3f} "
              f"{ev.device_time_total/1000:14.3f}")
    print("\nTop raw operators (self CPU)")
    print(p.key_averages().table(sort_by="self_cpu_time_total",row_limit=18))
    if trace:
        p.export_chrome_trace(str(trace))
        print("Chrome trace:",trace)
    if csv_path:
        with open(csv_path,"w",newline="",encoding="utf-8") as fh:
            writer=csv.writer(fh)
            writer.writerow(["function","calls","self_cpu_ms","cpu_total_ms",
                             "self_cuda_ms","cuda_total_ms"])
            for ev in events:
                writer.writerow([ev.key[4:],ev.count,
                     ev.self_cpu_time_total/1000,ev.cpu_time_total/1000,
                     ev.self_device_time_total/1000,ev.device_time_total/1000])
        print("CSV report:",csv_path)
    return p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps",type=int,default=5)
    ap.add_argument("--backend",choices=["research","sdpa"],default="research")
    ap.add_argument("--trace",default="function_profile_trace.json")
    ap.add_argument("--csv",default="function_profile_report.csv")
    a=ap.parse_args()
    profile_functions(a.device,a.steps,a.backend,a.trace,a.csv)

if __name__=="__main__":
    main()
