import pytest
import torch
from perfkit.train.losses import (
    causal_token_loss,per_example_losses,masked_accuracy,topk_accuracy)

@pytest.mark.parametrize('batch,seq,vocab',[(1,2,8),(3,7,13),(2,9,16)])
def test_loss_matches_manual_crossentropy(batch,seq,vocab):
    torch.manual_seed(batch+seq+vocab)
    logits=torch.randn(batch,seq,vocab)
    labels=torch.randint(0,vocab,(batch,seq))
    expected=torch.nn.functional.cross_entropy(logits.reshape(-1,vocab),labels.reshape(-1))
    torch.testing.assert_close(causal_token_loss(logits,labels),expected)

def test_ignore_labels_do_not_contribute():
    logits=torch.randn(2,4,7)
    targets=torch.tensor([[1,2,-100,-100],[3,4,5,-100]])
    output=per_example_losses(logits,targets)
    assert output.shape==(2,)
    assert torch.isfinite(output).all()

def test_accuracy_constructed_perfect():
    logits=torch.eye(8).unsqueeze(0)
    labels=torch.arange(8).unsqueeze(0)
    assert masked_accuracy(logits,labels)==1
    assert topk_accuracy(logits,labels,k=1)==1

def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):causal_token_loss(torch.randn(2,3,7),torch.ones(3,2,dtype=torch.long))
