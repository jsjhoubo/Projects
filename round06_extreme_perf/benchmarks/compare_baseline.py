"""Differential comparison for attention backends on shared weights/inputs."""
import argparse
import json
import torch
from perfkit.model.decoder import ResearchDecoder
from perfkit.infra.config import ModelConfig
from perfkit.eval.metrics import compare_logits


def compare(seq=32,batch=2,vocab=96,width=32,heads=4):
    cfg=ModelConfig(vocab_size=vocab,hidden_size=width,num_heads=heads,
                    num_layers=1,intermediate_size=width*2,
                    max_position_embeddings=max(seq,64),dropout=0.0)
    torch.manual_seed(31)
    model=ResearchDecoder(cfg)
    tokens=torch.randint(0,vocab,(batch,seq))
    baseline=model(tokens,'research')
    optimized=model(tokens,'sdpa')
    forward=compare_logits(baseline,optimized)
    torch.autograd.grad(baseline.sum(),tuple(model.parameters()),retain_graph=True)
    grads_a=torch.autograd.grad(baseline.square().sum(),tuple(model.parameters()),
                               retain_graph=True,allow_unused=True)
    grads_b=torch.autograd.grad(optimized.square().sum(),tuple(model.parameters()),
                               allow_unused=True)
    maxdiff=max(float((a-b).abs().max()) for a,b in zip(grads_a,grads_b)
                if a is not None and b is not None)
    return {'forward':forward,'max_gradient_difference':maxdiff,
            'batch':batch,'seq':seq,'dtype':str(tokens.dtype)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seq',type=int,default=32)
    p.add_argument('--batch',type=int,default=2)
    args=p.parse_args()
    print(json.dumps(compare(args.seq,args.batch),indent=2))

if __name__=='__main__':main()
