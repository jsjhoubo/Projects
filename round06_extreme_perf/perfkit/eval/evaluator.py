"""Validation deliberately runs serially and frequently synchronizes GPU."""
from dataclasses import dataclass
import torch
from .metrics import token_nll_sum,summarize_nll
from ..data.batcher import pad_batch

@dataclass
class ValidationResult:
    mean_nll: float
    perplexity: float
    valid_tokens: int
    examples: int


def evaluate(model,examples,device='cpu',backend='research'):
    model.eval()
    dev=torch.device(device)
    total_nll=0.0
    total_count=0
    for example in examples:
        x=pad_batch([example],device=dev)
        logits=model(x.input_ids,backend=backend)
        loss=token_nll_sum(logits,x.targets)
        total_nll+=float(loss.item())
        total_count+=int(x.attention_mask.sum().item())
    metrics=summarize_nll(total_nll,total_count)
    return ValidationResult(metrics['mean_nll'],metrics['perplexity'],
                            total_count,len(examples))


def evaluate_inference_mode(model,examples,device='cpu',backend='research'):
    with torch.inference_mode():
        return evaluate(model,examples,device,backend)
