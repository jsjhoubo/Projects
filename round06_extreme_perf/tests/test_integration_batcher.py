import pytest
import torch
from perfkit.data.corpus import synthetic_corpus
from perfkit.data.batcher import pad_batch,unpad_targets,make_microbatches

@pytest.mark.parametrize('batch',[1,2,7])
def test_padding_preserves_sequences(batch):
    records=synthetic_corpus(batch,min_length=4,max_length=16)
    result=pad_batch(records)
    assert result.input_ids.shape[0]==batch
    assert result.targets.shape==result.input_ids.shape
    assert result.attention_mask.shape==result.input_ids.shape
    for i,r in enumerate(records):
        width=r.length-1
        assert result.input_ids[i,:width].tolist()==list(r.tokens[:-1])
        assert result.targets[i,:width].tolist()==list(r.tokens[1:])
        assert bool(result.attention_mask[i,:width].all())


def test_padding_label_ignore_index():
    records=synthetic_corpus(40,min_length=4,max_length=16)
    batch=pad_batch(records)
    assert torch.equal(batch.targets[~batch.attention_mask],
                       torch.full_like(batch.targets[~batch.attention_mask],-100))

def test_batch_transfer_cpu_and_token_count():
    records=synthetic_corpus(4,min_length=4,max_length=10)
    batch=pad_batch(records)
    copied=batch.to('cpu')
    torch.testing.assert_close(copied.input_ids,batch.input_ids)
    assert copied.useful_tokens==sum(r.length-1 for r in records)

def test_microbatches_cover_all_records():
    records=synthetic_corpus(17)
    micros=list(make_microbatches(records,4))
    ids=[i for b in micros for i in b.ids]
    assert ids==list(range(17))


def test_empty_batch_rejected():
    with pytest.raises(ValueError):pad_batch([])
