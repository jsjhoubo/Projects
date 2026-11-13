"""Optimizer and gradient diagnostic utilities."""
import torch


def make_optimizer(model,lr=3e-4,weight_decay=0.1,algorithm='adamw'):
    groups=model.param_groups(weight_decay) if hasattr(model,'param_groups') else model.parameters()
    if algorithm=='adamw':return torch.optim.AdamW(groups,lr=lr)
    if algorithm=='sgd':return torch.optim.SGD(groups,lr=lr,momentum=.9)
    raise ValueError('optimizer')


def global_grad_norm(model):
    grads=[p.grad.detach().float().norm(2) for p in model.parameters()
           if p.grad is not None]
    if not grads:return torch.tensor(0.)
    return torch.stack(grads).norm(2)


def inspect_gradients(model):
    output={}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        g=p.grad.detach()
        output[name]={'mean_abs':float(g.abs().mean().item()),
                      'max_abs':float(g.abs().max().item()),
                      'finite':bool(torch.isfinite(g).all().item())}
    return output


def clip_gradients(model,max_norm):
    if max_norm<=0:return global_grad_norm(model)
    return torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm)


def params_equal(model_a,model_b,rtol=1e-5,atol=1e-6):
    pa=dict(model_a.named_parameters());pb=dict(model_b.named_parameters())
    if set(pa)!=set(pb):return False
    return all(torch.allclose(pa[key],pb[key],rtol=rtol,atol=atol) for key in pa)
