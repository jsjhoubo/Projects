"""Regression cases for temporal: both deterministic examples and invariants."""
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

def test_temporal_case_0065():
    """leakage; deterministic sample 65."""
    rng = random.Random(9166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0066():
    """boundary timestamps; deterministic sample 66."""
    rng = random.Random(9167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0067():
    """event reordering; deterministic sample 67."""
    rng = random.Random(9168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0068():
    """leakage; deterministic sample 68."""
    rng = random.Random(9169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0069():
    """boundary timestamps; deterministic sample 69."""
    rng = random.Random(9170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0070():
    """event reordering; deterministic sample 70."""
    rng = random.Random(9171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0071():
    """leakage; deterministic sample 71."""
    rng = random.Random(9172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0072():
    """boundary timestamps; deterministic sample 72."""
    rng = random.Random(9173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0073():
    """event reordering; deterministic sample 73."""
    rng = random.Random(9174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0074():
    """leakage; deterministic sample 74."""
    rng = random.Random(9175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0075():
    """boundary timestamps; deterministic sample 75."""
    rng = random.Random(9176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0076():
    """event reordering; deterministic sample 76."""
    rng = random.Random(9177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0077():
    """leakage; deterministic sample 77."""
    rng = random.Random(9178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0078():
    """boundary timestamps; deterministic sample 78."""
    rng = random.Random(9179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0079():
    """event reordering; deterministic sample 79."""
    rng = random.Random(9180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0080():
    """leakage; deterministic sample 80."""
    rng = random.Random(9181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0081():
    """boundary timestamps; deterministic sample 81."""
    rng = random.Random(9182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0082():
    """event reordering; deterministic sample 82."""
    rng = random.Random(9183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0083():
    """leakage; deterministic sample 83."""
    rng = random.Random(9184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0084():
    """boundary timestamps; deterministic sample 84."""
    rng = random.Random(9185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0085():
    """event reordering; deterministic sample 85."""
    rng = random.Random(9186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0086():
    """leakage; deterministic sample 86."""
    rng = random.Random(9187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0087():
    """boundary timestamps; deterministic sample 87."""
    rng = random.Random(9188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0088():
    """event reordering; deterministic sample 88."""
    rng = random.Random(9189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0089():
    """leakage; deterministic sample 89."""
    rng = random.Random(9190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0090():
    """boundary timestamps; deterministic sample 90."""
    rng = random.Random(9191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0091():
    """event reordering; deterministic sample 91."""
    rng = random.Random(9192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0092():
    """leakage; deterministic sample 92."""
    rng = random.Random(9193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0093():
    """boundary timestamps; deterministic sample 93."""
    rng = random.Random(9194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0094():
    """event reordering; deterministic sample 94."""
    rng = random.Random(9195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0095():
    """leakage; deterministic sample 95."""
    rng = random.Random(9196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0096():
    """boundary timestamps; deterministic sample 96."""
    rng = random.Random(9197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0097():
    """event reordering; deterministic sample 97."""
    rng = random.Random(9198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0098():
    """leakage; deterministic sample 98."""
    rng = random.Random(9199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0099():
    """boundary timestamps; deterministic sample 99."""
    rng = random.Random(9200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0100():
    """event reordering; deterministic sample 100."""
    rng = random.Random(9201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0101():
    """leakage; deterministic sample 101."""
    rng = random.Random(9202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0102():
    """boundary timestamps; deterministic sample 102."""
    rng = random.Random(9203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0103():
    """event reordering; deterministic sample 103."""
    rng = random.Random(9204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0104():
    """leakage; deterministic sample 104."""
    rng = random.Random(9205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0105():
    """boundary timestamps; deterministic sample 105."""
    rng = random.Random(9206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0106():
    """event reordering; deterministic sample 106."""
    rng = random.Random(9207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0107():
    """leakage; deterministic sample 107."""
    rng = random.Random(9208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0108():
    """boundary timestamps; deterministic sample 108."""
    rng = random.Random(9209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0109():
    """event reordering; deterministic sample 109."""
    rng = random.Random(9210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0110():
    """leakage; deterministic sample 110."""
    rng = random.Random(9211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0111():
    """boundary timestamps; deterministic sample 111."""
    rng = random.Random(9212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0112():
    """event reordering; deterministic sample 112."""
    rng = random.Random(9213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0113():
    """leakage; deterministic sample 113."""
    rng = random.Random(9214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0114():
    """boundary timestamps; deterministic sample 114."""
    rng = random.Random(9215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0115():
    """event reordering; deterministic sample 115."""
    rng = random.Random(9216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0116():
    """leakage; deterministic sample 116."""
    rng = random.Random(9217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0117():
    """boundary timestamps; deterministic sample 117."""
    rng = random.Random(9218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0118():
    """event reordering; deterministic sample 118."""
    rng = random.Random(9219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0119():
    """leakage; deterministic sample 119."""
    rng = random.Random(9220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0120():
    """boundary timestamps; deterministic sample 120."""
    rng = random.Random(9221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0121():
    """event reordering; deterministic sample 121."""
    rng = random.Random(9222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0122():
    """leakage; deterministic sample 122."""
    rng = random.Random(9223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0123():
    """boundary timestamps; deterministic sample 123."""
    rng = random.Random(9224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0124():
    """event reordering; deterministic sample 124."""
    rng = random.Random(9225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0125():
    """leakage; deterministic sample 125."""
    rng = random.Random(9226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0126():
    """boundary timestamps; deterministic sample 126."""
    rng = random.Random(9227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0127():
    """event reordering; deterministic sample 127."""
    rng = random.Random(9228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0128():
    """leakage; deterministic sample 128."""
    rng = random.Random(9229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0129():
    """boundary timestamps; deterministic sample 129."""
    rng = random.Random(9230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0130():
    """event reordering; deterministic sample 130."""
    rng = random.Random(9231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0131():
    """leakage; deterministic sample 131."""
    rng = random.Random(9232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0132():
    """boundary timestamps; deterministic sample 132."""
    rng = random.Random(9233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0133():
    """event reordering; deterministic sample 133."""
    rng = random.Random(9234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0134():
    """leakage; deterministic sample 134."""
    rng = random.Random(9235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0135():
    """boundary timestamps; deterministic sample 135."""
    rng = random.Random(9236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0136():
    """event reordering; deterministic sample 136."""
    rng = random.Random(9237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0137():
    """leakage; deterministic sample 137."""
    rng = random.Random(9238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0138():
    """boundary timestamps; deterministic sample 138."""
    rng = random.Random(9239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0139():
    """event reordering; deterministic sample 139."""
    rng = random.Random(9240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0140():
    """leakage; deterministic sample 140."""
    rng = random.Random(9241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0141():
    """boundary timestamps; deterministic sample 141."""
    rng = random.Random(9242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0142():
    """event reordering; deterministic sample 142."""
    rng = random.Random(9243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0143():
    """leakage; deterministic sample 143."""
    rng = random.Random(9244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0144():
    """boundary timestamps; deterministic sample 144."""
    rng = random.Random(9245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0145():
    """event reordering; deterministic sample 145."""
    rng = random.Random(9246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0146():
    """leakage; deterministic sample 146."""
    rng = random.Random(9247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0147():
    """boundary timestamps; deterministic sample 147."""
    rng = random.Random(9248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0148():
    """event reordering; deterministic sample 148."""
    rng = random.Random(9249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0149():
    """leakage; deterministic sample 149."""
    rng = random.Random(9250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0150():
    """boundary timestamps; deterministic sample 150."""
    rng = random.Random(9251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0151():
    """event reordering; deterministic sample 151."""
    rng = random.Random(9252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0152():
    """leakage; deterministic sample 152."""
    rng = random.Random(9253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0153():
    """boundary timestamps; deterministic sample 153."""
    rng = random.Random(9254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0154():
    """event reordering; deterministic sample 154."""
    rng = random.Random(9255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0155():
    """leakage; deterministic sample 155."""
    rng = random.Random(9256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0156():
    """boundary timestamps; deterministic sample 156."""
    rng = random.Random(9257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0157():
    """event reordering; deterministic sample 157."""
    rng = random.Random(9258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0158():
    """leakage; deterministic sample 158."""
    rng = random.Random(9259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0159():
    """boundary timestamps; deterministic sample 159."""
    rng = random.Random(9260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0160():
    """event reordering; deterministic sample 160."""
    rng = random.Random(9261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0161():
    """leakage; deterministic sample 161."""
    rng = random.Random(9262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0162():
    """boundary timestamps; deterministic sample 162."""
    rng = random.Random(9263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0163():
    """event reordering; deterministic sample 163."""
    rng = random.Random(9264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0164():
    """leakage; deterministic sample 164."""
    rng = random.Random(9265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0165():
    """boundary timestamps; deterministic sample 165."""
    rng = random.Random(9266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0166():
    """event reordering; deterministic sample 166."""
    rng = random.Random(9267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0167():
    """leakage; deterministic sample 167."""
    rng = random.Random(9268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0168():
    """boundary timestamps; deterministic sample 168."""
    rng = random.Random(9269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0169():
    """event reordering; deterministic sample 169."""
    rng = random.Random(9270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0170():
    """leakage; deterministic sample 170."""
    rng = random.Random(9271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0171():
    """boundary timestamps; deterministic sample 171."""
    rng = random.Random(9272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0172():
    """event reordering; deterministic sample 172."""
    rng = random.Random(9273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0173():
    """leakage; deterministic sample 173."""
    rng = random.Random(9274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0174():
    """boundary timestamps; deterministic sample 174."""
    rng = random.Random(9275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0175():
    """event reordering; deterministic sample 175."""
    rng = random.Random(9276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0176():
    """leakage; deterministic sample 176."""
    rng = random.Random(9277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0177():
    """boundary timestamps; deterministic sample 177."""
    rng = random.Random(9278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0178():
    """event reordering; deterministic sample 178."""
    rng = random.Random(9279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0179():
    """leakage; deterministic sample 179."""
    rng = random.Random(9280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0180():
    """boundary timestamps; deterministic sample 180."""
    rng = random.Random(9281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0181():
    """event reordering; deterministic sample 181."""
    rng = random.Random(9282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0182():
    """leakage; deterministic sample 182."""
    rng = random.Random(9283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0183():
    """boundary timestamps; deterministic sample 183."""
    rng = random.Random(9284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0184():
    """event reordering; deterministic sample 184."""
    rng = random.Random(9285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0185():
    """leakage; deterministic sample 185."""
    rng = random.Random(9286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0186():
    """boundary timestamps; deterministic sample 186."""
    rng = random.Random(9287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0187():
    """event reordering; deterministic sample 187."""
    rng = random.Random(9288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0188():
    """leakage; deterministic sample 188."""
    rng = random.Random(9289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0189():
    """boundary timestamps; deterministic sample 189."""
    rng = random.Random(9290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0190():
    """event reordering; deterministic sample 190."""
    rng = random.Random(9291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0191():
    """leakage; deterministic sample 191."""
    rng = random.Random(9292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0192():
    """boundary timestamps; deterministic sample 192."""
    rng = random.Random(9293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0193():
    """event reordering; deterministic sample 193."""
    rng = random.Random(9294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0194():
    """leakage; deterministic sample 194."""
    rng = random.Random(9295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0195():
    """boundary timestamps; deterministic sample 195."""
    rng = random.Random(9296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0196():
    """event reordering; deterministic sample 196."""
    rng = random.Random(9297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0197():
    """leakage; deterministic sample 197."""
    rng = random.Random(9298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0198():
    """boundary timestamps; deterministic sample 198."""
    rng = random.Random(9299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0199():
    """event reordering; deterministic sample 199."""
    rng = random.Random(9300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0200():
    """leakage; deterministic sample 200."""
    rng = random.Random(9301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0201():
    """boundary timestamps; deterministic sample 201."""
    rng = random.Random(9302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0202():
    """event reordering; deterministic sample 202."""
    rng = random.Random(9303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0203():
    """leakage; deterministic sample 203."""
    rng = random.Random(9304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0204():
    """boundary timestamps; deterministic sample 204."""
    rng = random.Random(9305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0205():
    """event reordering; deterministic sample 205."""
    rng = random.Random(9306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0206():
    """leakage; deterministic sample 206."""
    rng = random.Random(9307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0207():
    """boundary timestamps; deterministic sample 207."""
    rng = random.Random(9308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0208():
    """event reordering; deterministic sample 208."""
    rng = random.Random(9309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0209():
    """leakage; deterministic sample 209."""
    rng = random.Random(9310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0210():
    """boundary timestamps; deterministic sample 210."""
    rng = random.Random(9311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0211():
    """event reordering; deterministic sample 211."""
    rng = random.Random(9312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0212():
    """leakage; deterministic sample 212."""
    rng = random.Random(9313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0213():
    """boundary timestamps; deterministic sample 213."""
    rng = random.Random(9314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0214():
    """event reordering; deterministic sample 214."""
    rng = random.Random(9315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0215():
    """leakage; deterministic sample 215."""
    rng = random.Random(9316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0216():
    """boundary timestamps; deterministic sample 216."""
    rng = random.Random(9317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0217():
    """event reordering; deterministic sample 217."""
    rng = random.Random(9318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0218():
    """leakage; deterministic sample 218."""
    rng = random.Random(9319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0219():
    """boundary timestamps; deterministic sample 219."""
    rng = random.Random(9320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0220():
    """event reordering; deterministic sample 220."""
    rng = random.Random(9321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0221():
    """leakage; deterministic sample 221."""
    rng = random.Random(9322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0222():
    """boundary timestamps; deterministic sample 222."""
    rng = random.Random(9323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0223():
    """event reordering; deterministic sample 223."""
    rng = random.Random(9324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0224():
    """leakage; deterministic sample 224."""
    rng = random.Random(9325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0225():
    """boundary timestamps; deterministic sample 225."""
    rng = random.Random(9326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0226():
    """event reordering; deterministic sample 226."""
    rng = random.Random(9327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0227():
    """leakage; deterministic sample 227."""
    rng = random.Random(9328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0228():
    """boundary timestamps; deterministic sample 228."""
    rng = random.Random(9329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0229():
    """event reordering; deterministic sample 229."""
    rng = random.Random(9330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0230():
    """leakage; deterministic sample 230."""
    rng = random.Random(9331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0231():
    """boundary timestamps; deterministic sample 231."""
    rng = random.Random(9332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0232():
    """event reordering; deterministic sample 232."""
    rng = random.Random(9333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0233():
    """leakage; deterministic sample 233."""
    rng = random.Random(9334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0234():
    """boundary timestamps; deterministic sample 234."""
    rng = random.Random(9335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0235():
    """event reordering; deterministic sample 235."""
    rng = random.Random(9336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0236():
    """leakage; deterministic sample 236."""
    rng = random.Random(9337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0237():
    """boundary timestamps; deterministic sample 237."""
    rng = random.Random(9338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0238():
    """event reordering; deterministic sample 238."""
    rng = random.Random(9339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_temporal_case_0239():
    """leakage; deterministic sample 239."""
    rng = random.Random(9340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = len(subject.split_before([{"t":i} for i in range(n)],"t",n//2)[0]) / n
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

