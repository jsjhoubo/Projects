from collections import Counter
import pytest
from perfkit.data.corpus import (
    synthetic_corpus,split_by_time,split_by_user,corpus_histogram,verify_no_overlap)

@pytest.mark.parametrize('n',[0,1,8,100])
def test_deterministic_corpus(n):
    a=synthetic_corpus(n,seed=11,users=4)
    b=synthetic_corpus(n,seed=11,users=4)
    assert a==b
    assert len(a)==n

@pytest.mark.parametrize('skew',[0,.5,1.0])
def test_corpus_lengths_and_token_boundaries(skew):
    records=synthetic_corpus(64,min_length=5,max_length=20,vocab=70,skew=skew)
    assert all(5<=e.length<=20 for e in records)
    assert all(1<=t<70 for e in records for t in e.tokens)

def test_user_group_split_no_overlap():
    records=synthetic_corpus(200,users=10)
    train,val=split_by_user(records,[0,1,2])
    assert verify_no_overlap(train,val)
    assert {e.user_id for e in train}.isdisjoint({e.user_id for e in val})

def test_time_split_boundary_is_half_open():
    records=synthetic_corpus(20)
    train,val=split_by_time(records,7*7)
    assert all(e.timestamp<49 for e in train)
    assert all(e.timestamp>=49 for e in val)
    assert verify_no_overlap(train,val)

def test_histogram_reconciles_total():
    records=synthetic_corpus(64)
    stats=corpus_histogram(records)
    assert sum(stats['lengths'].values())==64
    assert sum(stats['users'].values())==64
    assert sum(stats['tokens'].values())==sum(e.length for e in records)
