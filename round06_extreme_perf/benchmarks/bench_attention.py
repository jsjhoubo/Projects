"""Controlled microbenchmark; report forward AND backward and memory."""
import argparse
import statistics
import torch
from perfkit.model.attention import ResearchMultiheadAttention
from perfkit.infra.timing import section,Timings,reset_peak,peak_allocated


def attention_profile(batch=4,sequence=64,hidden=64,heads=4,
                      backend='research',device='cpu',repeats=5):
    torch.manual_seed(19)
    module=ResearchMultiheadAttention(hidden,heads,dropout=0.0).to(device)
    x=torch.randn(batch,sequence,hidden,device=device,requires_grad=True)
    timing=Timings()
    for i in range(repeats+2):
        module.zero_grad(set_to_none=True)
        x.grad=None
        reset_peak(device)
        measurement=Timings() if i<2 else timing
        with section('forward',measurement,device):
            y=module(x,backend=backend)
        with section('backward',measurement,device):
            y.square().mean().backward()
    return {'batch':batch,'sequence':sequence,'heads':heads,
            'backend':backend,'device':device,
            'timings':timing.summary(),
            'peak_memory_bytes':peak_allocated(device),
            'last_grad_max':float(x.grad.detach().abs().amax())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seq',type=int,default=64)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--backend',choices=['research','sdpa'],default='research')
    args=parser.parse_args()
    print(attention_profile(args.batch,args.seq,backend=args.backend,
                            device=args.device))

if __name__=='__main__':main()
