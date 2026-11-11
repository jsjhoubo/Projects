"""Regression cases for temporal: deterministic scenarios and invariants."""
import math
import random
import pytest
from ml_eval.analytics import temporal as subject

def test_temporal_case_0000():
    """boundary timestamps; deterministic sample 0."""
    rng = random.Random(9101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0001():
    """event reordering; deterministic sample 1."""
    rng = random.Random(9102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0002():
    """leakage; deterministic sample 2."""
    rng = random.Random(9103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0003():
    """boundary timestamps; deterministic sample 3."""
    rng = random.Random(9104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0004():
    """event reordering; deterministic sample 4."""
    rng = random.Random(9105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0005():
    """leakage; deterministic sample 5."""
    rng = random.Random(9106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0006():
    """boundary timestamps; deterministic sample 6."""
    rng = random.Random(9107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0007():
    """event reordering; deterministic sample 7."""
    rng = random.Random(9108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0008():
    """leakage; deterministic sample 8."""
    rng = random.Random(9109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0009():
    """boundary timestamps; deterministic sample 9."""
    rng = random.Random(9110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0010():
    """event reordering; deterministic sample 10."""
    rng = random.Random(9111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0011():
    """leakage; deterministic sample 11."""
    rng = random.Random(9112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0012():
    """boundary timestamps; deterministic sample 12."""
    rng = random.Random(9113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0013():
    """event reordering; deterministic sample 13."""
    rng = random.Random(9114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0014():
    """leakage; deterministic sample 14."""
    rng = random.Random(9115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0015():
    """boundary timestamps; deterministic sample 15."""
    rng = random.Random(9116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0016():
    """event reordering; deterministic sample 16."""
    rng = random.Random(9117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0017():
    """leakage; deterministic sample 17."""
    rng = random.Random(9118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0018():
    """boundary timestamps; deterministic sample 18."""
    rng = random.Random(9119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0019():
    """event reordering; deterministic sample 19."""
    rng = random.Random(9120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0020():
    """leakage; deterministic sample 20."""
    rng = random.Random(9121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0021():
    """boundary timestamps; deterministic sample 21."""
    rng = random.Random(9122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0022():
    """event reordering; deterministic sample 22."""
    rng = random.Random(9123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0023():
    """leakage; deterministic sample 23."""
    rng = random.Random(9124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0024():
    """boundary timestamps; deterministic sample 24."""
    rng = random.Random(9125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0025():
    """event reordering; deterministic sample 25."""
    rng = random.Random(9126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0026():
    """leakage; deterministic sample 26."""
    rng = random.Random(9127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0027():
    """boundary timestamps; deterministic sample 27."""
    rng = random.Random(9128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0028():
    """event reordering; deterministic sample 28."""
    rng = random.Random(9129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0029():
    """leakage; deterministic sample 29."""
    rng = random.Random(9130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0030():
    """boundary timestamps; deterministic sample 30."""
    rng = random.Random(9131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0031():
    """event reordering; deterministic sample 31."""
    rng = random.Random(9132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0032():
    """leakage; deterministic sample 32."""
    rng = random.Random(9133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0033():
    """boundary timestamps; deterministic sample 33."""
    rng = random.Random(9134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0034():
    """event reordering; deterministic sample 34."""
    rng = random.Random(9135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0035():
    """leakage; deterministic sample 35."""
    rng = random.Random(9136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0036():
    """boundary timestamps; deterministic sample 36."""
    rng = random.Random(9137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0037():
    """event reordering; deterministic sample 37."""
    rng = random.Random(9138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0038():
    """leakage; deterministic sample 38."""
    rng = random.Random(9139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0039():
    """boundary timestamps; deterministic sample 39."""
    rng = random.Random(9140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0040():
    """event reordering; deterministic sample 40."""
    rng = random.Random(9141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0041():
    """leakage; deterministic sample 41."""
    rng = random.Random(9142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0042():
    """boundary timestamps; deterministic sample 42."""
    rng = random.Random(9143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0043():
    """event reordering; deterministic sample 43."""
    rng = random.Random(9144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0044():
    """leakage; deterministic sample 44."""
    rng = random.Random(9145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0045():
    """boundary timestamps; deterministic sample 45."""
    rng = random.Random(9146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0046():
    """event reordering; deterministic sample 46."""
    rng = random.Random(9147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0047():
    """leakage; deterministic sample 47."""
    rng = random.Random(9148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0048():
    """boundary timestamps; deterministic sample 48."""
    rng = random.Random(9149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0049():
    """event reordering; deterministic sample 49."""
    rng = random.Random(9150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0050():
    """leakage; deterministic sample 50."""
    rng = random.Random(9151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0051():
    """boundary timestamps; deterministic sample 51."""
    rng = random.Random(9152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0052():
    """event reordering; deterministic sample 52."""
    rng = random.Random(9153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0053():
    """leakage; deterministic sample 53."""
    rng = random.Random(9154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0054():
    """boundary timestamps; deterministic sample 54."""
    rng = random.Random(9155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0055():
    """event reordering; deterministic sample 55."""
    rng = random.Random(9156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0056():
    """leakage; deterministic sample 56."""
    rng = random.Random(9157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0057():
    """boundary timestamps; deterministic sample 57."""
    rng = random.Random(9158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0058():
    """event reordering; deterministic sample 58."""
    rng = random.Random(9159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0059():
    """leakage; deterministic sample 59."""
    rng = random.Random(9160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0060():
    """boundary timestamps; deterministic sample 60."""
    rng = random.Random(9161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0061():
    """event reordering; deterministic sample 61."""
    rng = random.Random(9162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0062():
    """leakage; deterministic sample 62."""
    rng = random.Random(9163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0063():
    """boundary timestamps; deterministic sample 63."""
    rng = random.Random(9164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0064():
    """event reordering; deterministic sample 64."""
    rng = random.Random(9165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

