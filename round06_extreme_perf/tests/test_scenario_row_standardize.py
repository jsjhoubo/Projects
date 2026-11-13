"""Correctness gates for zscore_kernel before kernel-level performance changes."""
import pytest
import torch
from scenarios.row_standardize import (
    research_op,repeated_columns,compare_performance,runtime_stress,
    gradient_sample
)


def independent_oracle(x,y,scale):
    """Small PyTorch reference in a different expression/path."""
    return torch.nn.functional.layer_norm(x*scale,(x.size(-1),),eps=1e-4)


@pytest.mark.parametrize('batch,width',[(1,8),(2,16),(8,32),(17,64)])
@pytest.mark.parametrize('scale',[0.5,1.0,2.0])
def test_reference_row_standardize(batch,width,scale):
    torch.manual_seed(batch+width)
    x=torch.randn(batch,width,dtype=torch.float32)
    y=torch.randn(batch,width,dtype=torch.float32)
    got=research_op(x,y,scale)
    expected=independent_oracle(x,y,scale)
    torch.testing.assert_close(got,expected,rtol=2e-4,atol=2e-5)


def test_nonfinite_row_standardize_rejected():
    x=torch.zeros(2,8)
    x[0,0]=float('nan')
    with pytest.raises(ValueError):
        research_op(x,torch.zeros_like(x))


def test_invalid_scale_row_standardize():
    x=torch.randn(2,8)
    with pytest.raises(ValueError):research_op(x,scale=-1)
    with pytest.raises(ValueError):research_op(x,scale=0)


def test_report_row_standardize_shape_and_time():
    x,y=runtime_stress(4,16)
    report=compare_performance(x,y,repeats=2)
    assert report['median_seconds']>=0
    assert report['min_seconds']<=report['max_seconds']
    assert report['input_dtype']=='torch.float32'
    assert report['total_elements']==64


def test_grad_row_standardize_finite():
    x,y=runtime_stress(4,16)
    grad=gradient_sample(x,y,1.0)
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.shape==x.shape


def test_deterministic_row_standardize():
    x,y=runtime_stress(4,16,seed=42)
    a=research_op(x,y)
    b=research_op(x,y)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_input_not_mutated_row_standardize():
    x,y=runtime_stress(3,16)
    old_x,old_y=x.detach().clone(),y.clone()
    research_op(x,y)
    torch.testing.assert_close(x.detach(),old_x)
    torch.testing.assert_close(y,old_y)
