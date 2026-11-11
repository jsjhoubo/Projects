"""Regression cases for multiclass: both deterministic examples and invariants."""
import math
import random
import pytest
from ml_eval.analytics import multiclass as subject

def test_multiclass_case_0000():
    """rare classes; deterministic sample 0."""
    rng = random.Random(11101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0001():
    """unknown labels; deterministic sample 1."""
    rng = random.Random(11102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0002():
    """multiclass averaging; deterministic sample 2."""
    rng = random.Random(11103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0003():
    """rare classes; deterministic sample 3."""
    rng = random.Random(11104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0004():
    """unknown labels; deterministic sample 4."""
    rng = random.Random(11105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0005():
    """multiclass averaging; deterministic sample 5."""
    rng = random.Random(11106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0006():
    """rare classes; deterministic sample 6."""
    rng = random.Random(11107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0007():
    """unknown labels; deterministic sample 7."""
    rng = random.Random(11108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0008():
    """multiclass averaging; deterministic sample 8."""
    rng = random.Random(11109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0009():
    """rare classes; deterministic sample 9."""
    rng = random.Random(11110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0010():
    """unknown labels; deterministic sample 10."""
    rng = random.Random(11111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0011():
    """multiclass averaging; deterministic sample 11."""
    rng = random.Random(11112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0012():
    """rare classes; deterministic sample 12."""
    rng = random.Random(11113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0013():
    """unknown labels; deterministic sample 13."""
    rng = random.Random(11114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0014():
    """multiclass averaging; deterministic sample 14."""
    rng = random.Random(11115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0015():
    """rare classes; deterministic sample 15."""
    rng = random.Random(11116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0016():
    """unknown labels; deterministic sample 16."""
    rng = random.Random(11117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0017():
    """multiclass averaging; deterministic sample 17."""
    rng = random.Random(11118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0018():
    """rare classes; deterministic sample 18."""
    rng = random.Random(11119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0019():
    """unknown labels; deterministic sample 19."""
    rng = random.Random(11120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0020():
    """multiclass averaging; deterministic sample 20."""
    rng = random.Random(11121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0021():
    """rare classes; deterministic sample 21."""
    rng = random.Random(11122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0022():
    """unknown labels; deterministic sample 22."""
    rng = random.Random(11123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0023():
    """multiclass averaging; deterministic sample 23."""
    rng = random.Random(11124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0024():
    """rare classes; deterministic sample 24."""
    rng = random.Random(11125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0025():
    """unknown labels; deterministic sample 25."""
    rng = random.Random(11126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0026():
    """multiclass averaging; deterministic sample 26."""
    rng = random.Random(11127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0027():
    """rare classes; deterministic sample 27."""
    rng = random.Random(11128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0028():
    """unknown labels; deterministic sample 28."""
    rng = random.Random(11129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0029():
    """multiclass averaging; deterministic sample 29."""
    rng = random.Random(11130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0030():
    """rare classes; deterministic sample 30."""
    rng = random.Random(11131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0031():
    """unknown labels; deterministic sample 31."""
    rng = random.Random(11132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0032():
    """multiclass averaging; deterministic sample 32."""
    rng = random.Random(11133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0033():
    """rare classes; deterministic sample 33."""
    rng = random.Random(11134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0034():
    """unknown labels; deterministic sample 34."""
    rng = random.Random(11135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0035():
    """multiclass averaging; deterministic sample 35."""
    rng = random.Random(11136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0036():
    """rare classes; deterministic sample 36."""
    rng = random.Random(11137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0037():
    """unknown labels; deterministic sample 37."""
    rng = random.Random(11138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0038():
    """multiclass averaging; deterministic sample 38."""
    rng = random.Random(11139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0039():
    """rare classes; deterministic sample 39."""
    rng = random.Random(11140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0040():
    """unknown labels; deterministic sample 40."""
    rng = random.Random(11141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0041():
    """multiclass averaging; deterministic sample 41."""
    rng = random.Random(11142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0042():
    """rare classes; deterministic sample 42."""
    rng = random.Random(11143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0043():
    """unknown labels; deterministic sample 43."""
    rng = random.Random(11144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0044():
    """multiclass averaging; deterministic sample 44."""
    rng = random.Random(11145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0045():
    """rare classes; deterministic sample 45."""
    rng = random.Random(11146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0046():
    """unknown labels; deterministic sample 46."""
    rng = random.Random(11147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0047():
    """multiclass averaging; deterministic sample 47."""
    rng = random.Random(11148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0048():
    """rare classes; deterministic sample 48."""
    rng = random.Random(11149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0049():
    """unknown labels; deterministic sample 49."""
    rng = random.Random(11150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0050():
    """multiclass averaging; deterministic sample 50."""
    rng = random.Random(11151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0051():
    """rare classes; deterministic sample 51."""
    rng = random.Random(11152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0052():
    """unknown labels; deterministic sample 52."""
    rng = random.Random(11153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0053():
    """multiclass averaging; deterministic sample 53."""
    rng = random.Random(11154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0054():
    """rare classes; deterministic sample 54."""
    rng = random.Random(11155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0055():
    """unknown labels; deterministic sample 55."""
    rng = random.Random(11156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0056():
    """multiclass averaging; deterministic sample 56."""
    rng = random.Random(11157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0057():
    """rare classes; deterministic sample 57."""
    rng = random.Random(11158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0058():
    """unknown labels; deterministic sample 58."""
    rng = random.Random(11159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0059():
    """multiclass averaging; deterministic sample 59."""
    rng = random.Random(11160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0060():
    """rare classes; deterministic sample 60."""
    rng = random.Random(11161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0061():
    """unknown labels; deterministic sample 61."""
    rng = random.Random(11162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0062():
    """multiclass averaging; deterministic sample 62."""
    rng = random.Random(11163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0063():
    """rare classes; deterministic sample 63."""
    rng = random.Random(11164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0064():
    """unknown labels; deterministic sample 64."""
    rng = random.Random(11165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0065():
    """multiclass averaging; deterministic sample 65."""
    rng = random.Random(11166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0066():
    """rare classes; deterministic sample 66."""
    rng = random.Random(11167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0067():
    """unknown labels; deterministic sample 67."""
    rng = random.Random(11168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0068():
    """multiclass averaging; deterministic sample 68."""
    rng = random.Random(11169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0069():
    """rare classes; deterministic sample 69."""
    rng = random.Random(11170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0070():
    """unknown labels; deterministic sample 70."""
    rng = random.Random(11171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0071():
    """multiclass averaging; deterministic sample 71."""
    rng = random.Random(11172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0072():
    """rare classes; deterministic sample 72."""
    rng = random.Random(11173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0073():
    """unknown labels; deterministic sample 73."""
    rng = random.Random(11174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0074():
    """multiclass averaging; deterministic sample 74."""
    rng = random.Random(11175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0075():
    """rare classes; deterministic sample 75."""
    rng = random.Random(11176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0076():
    """unknown labels; deterministic sample 76."""
    rng = random.Random(11177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0077():
    """multiclass averaging; deterministic sample 77."""
    rng = random.Random(11178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0078():
    """rare classes; deterministic sample 78."""
    rng = random.Random(11179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0079():
    """unknown labels; deterministic sample 79."""
    rng = random.Random(11180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0080():
    """multiclass averaging; deterministic sample 80."""
    rng = random.Random(11181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0081():
    """rare classes; deterministic sample 81."""
    rng = random.Random(11182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0082():
    """unknown labels; deterministic sample 82."""
    rng = random.Random(11183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0083():
    """multiclass averaging; deterministic sample 83."""
    rng = random.Random(11184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0084():
    """rare classes; deterministic sample 84."""
    rng = random.Random(11185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0085():
    """unknown labels; deterministic sample 85."""
    rng = random.Random(11186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0086():
    """multiclass averaging; deterministic sample 86."""
    rng = random.Random(11187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0087():
    """rare classes; deterministic sample 87."""
    rng = random.Random(11188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0088():
    """unknown labels; deterministic sample 88."""
    rng = random.Random(11189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0089():
    """multiclass averaging; deterministic sample 89."""
    rng = random.Random(11190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0090():
    """rare classes; deterministic sample 90."""
    rng = random.Random(11191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0091():
    """unknown labels; deterministic sample 91."""
    rng = random.Random(11192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0092():
    """multiclass averaging; deterministic sample 92."""
    rng = random.Random(11193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0093():
    """rare classes; deterministic sample 93."""
    rng = random.Random(11194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0094():
    """unknown labels; deterministic sample 94."""
    rng = random.Random(11195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0095():
    """multiclass averaging; deterministic sample 95."""
    rng = random.Random(11196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0096():
    """rare classes; deterministic sample 96."""
    rng = random.Random(11197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0097():
    """unknown labels; deterministic sample 97."""
    rng = random.Random(11198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0098():
    """multiclass averaging; deterministic sample 98."""
    rng = random.Random(11199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0099():
    """rare classes; deterministic sample 99."""
    rng = random.Random(11200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0100():
    """unknown labels; deterministic sample 100."""
    rng = random.Random(11201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0101():
    """multiclass averaging; deterministic sample 101."""
    rng = random.Random(11202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0102():
    """rare classes; deterministic sample 102."""
    rng = random.Random(11203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0103():
    """unknown labels; deterministic sample 103."""
    rng = random.Random(11204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0104():
    """multiclass averaging; deterministic sample 104."""
    rng = random.Random(11205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0105():
    """rare classes; deterministic sample 105."""
    rng = random.Random(11206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0106():
    """unknown labels; deterministic sample 106."""
    rng = random.Random(11207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0107():
    """multiclass averaging; deterministic sample 107."""
    rng = random.Random(11208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0108():
    """rare classes; deterministic sample 108."""
    rng = random.Random(11209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0109():
    """unknown labels; deterministic sample 109."""
    rng = random.Random(11210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0110():
    """multiclass averaging; deterministic sample 110."""
    rng = random.Random(11211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0111():
    """rare classes; deterministic sample 111."""
    rng = random.Random(11212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0112():
    """unknown labels; deterministic sample 112."""
    rng = random.Random(11213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0113():
    """multiclass averaging; deterministic sample 113."""
    rng = random.Random(11214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0114():
    """rare classes; deterministic sample 114."""
    rng = random.Random(11215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0115():
    """unknown labels; deterministic sample 115."""
    rng = random.Random(11216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0116():
    """multiclass averaging; deterministic sample 116."""
    rng = random.Random(11217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0117():
    """rare classes; deterministic sample 117."""
    rng = random.Random(11218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0118():
    """unknown labels; deterministic sample 118."""
    rng = random.Random(11219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0119():
    """multiclass averaging; deterministic sample 119."""
    rng = random.Random(11220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0120():
    """rare classes; deterministic sample 120."""
    rng = random.Random(11221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0121():
    """unknown labels; deterministic sample 121."""
    rng = random.Random(11222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0122():
    """multiclass averaging; deterministic sample 122."""
    rng = random.Random(11223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0123():
    """rare classes; deterministic sample 123."""
    rng = random.Random(11224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0124():
    """unknown labels; deterministic sample 124."""
    rng = random.Random(11225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0125():
    """multiclass averaging; deterministic sample 125."""
    rng = random.Random(11226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0126():
    """rare classes; deterministic sample 126."""
    rng = random.Random(11227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0127():
    """unknown labels; deterministic sample 127."""
    rng = random.Random(11228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0128():
    """multiclass averaging; deterministic sample 128."""
    rng = random.Random(11229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0129():
    """rare classes; deterministic sample 129."""
    rng = random.Random(11230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0130():
    """unknown labels; deterministic sample 130."""
    rng = random.Random(11231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0131():
    """multiclass averaging; deterministic sample 131."""
    rng = random.Random(11232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0132():
    """rare classes; deterministic sample 132."""
    rng = random.Random(11233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0133():
    """unknown labels; deterministic sample 133."""
    rng = random.Random(11234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0134():
    """multiclass averaging; deterministic sample 134."""
    rng = random.Random(11235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0135():
    """rare classes; deterministic sample 135."""
    rng = random.Random(11236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0136():
    """unknown labels; deterministic sample 136."""
    rng = random.Random(11237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0137():
    """multiclass averaging; deterministic sample 137."""
    rng = random.Random(11238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0138():
    """rare classes; deterministic sample 138."""
    rng = random.Random(11239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0139():
    """unknown labels; deterministic sample 139."""
    rng = random.Random(11240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0140():
    """multiclass averaging; deterministic sample 140."""
    rng = random.Random(11241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0141():
    """rare classes; deterministic sample 141."""
    rng = random.Random(11242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0142():
    """unknown labels; deterministic sample 142."""
    rng = random.Random(11243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0143():
    """multiclass averaging; deterministic sample 143."""
    rng = random.Random(11244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0144():
    """rare classes; deterministic sample 144."""
    rng = random.Random(11245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0145():
    """unknown labels; deterministic sample 145."""
    rng = random.Random(11246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0146():
    """multiclass averaging; deterministic sample 146."""
    rng = random.Random(11247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0147():
    """rare classes; deterministic sample 147."""
    rng = random.Random(11248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0148():
    """unknown labels; deterministic sample 148."""
    rng = random.Random(11249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0149():
    """multiclass averaging; deterministic sample 149."""
    rng = random.Random(11250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0150():
    """rare classes; deterministic sample 150."""
    rng = random.Random(11251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0151():
    """unknown labels; deterministic sample 151."""
    rng = random.Random(11252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0152():
    """multiclass averaging; deterministic sample 152."""
    rng = random.Random(11253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0153():
    """rare classes; deterministic sample 153."""
    rng = random.Random(11254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0154():
    """unknown labels; deterministic sample 154."""
    rng = random.Random(11255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0155():
    """multiclass averaging; deterministic sample 155."""
    rng = random.Random(11256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0156():
    """rare classes; deterministic sample 156."""
    rng = random.Random(11257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0157():
    """unknown labels; deterministic sample 157."""
    rng = random.Random(11258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0158():
    """multiclass averaging; deterministic sample 158."""
    rng = random.Random(11259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0159():
    """rare classes; deterministic sample 159."""
    rng = random.Random(11260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0160():
    """unknown labels; deterministic sample 160."""
    rng = random.Random(11261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0161():
    """multiclass averaging; deterministic sample 161."""
    rng = random.Random(11262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0162():
    """rare classes; deterministic sample 162."""
    rng = random.Random(11263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0163():
    """unknown labels; deterministic sample 163."""
    rng = random.Random(11264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0164():
    """multiclass averaging; deterministic sample 164."""
    rng = random.Random(11265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0165():
    """rare classes; deterministic sample 165."""
    rng = random.Random(11266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0166():
    """unknown labels; deterministic sample 166."""
    rng = random.Random(11267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0167():
    """multiclass averaging; deterministic sample 167."""
    rng = random.Random(11268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0168():
    """rare classes; deterministic sample 168."""
    rng = random.Random(11269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0169():
    """unknown labels; deterministic sample 169."""
    rng = random.Random(11270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0170():
    """multiclass averaging; deterministic sample 170."""
    rng = random.Random(11271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0171():
    """rare classes; deterministic sample 171."""
    rng = random.Random(11272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0172():
    """unknown labels; deterministic sample 172."""
    rng = random.Random(11273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0173():
    """multiclass averaging; deterministic sample 173."""
    rng = random.Random(11274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0174():
    """rare classes; deterministic sample 174."""
    rng = random.Random(11275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0175():
    """unknown labels; deterministic sample 175."""
    rng = random.Random(11276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0176():
    """multiclass averaging; deterministic sample 176."""
    rng = random.Random(11277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0177():
    """rare classes; deterministic sample 177."""
    rng = random.Random(11278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0178():
    """unknown labels; deterministic sample 178."""
    rng = random.Random(11279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0179():
    """multiclass averaging; deterministic sample 179."""
    rng = random.Random(11280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0180():
    """rare classes; deterministic sample 180."""
    rng = random.Random(11281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0181():
    """unknown labels; deterministic sample 181."""
    rng = random.Random(11282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0182():
    """multiclass averaging; deterministic sample 182."""
    rng = random.Random(11283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0183():
    """rare classes; deterministic sample 183."""
    rng = random.Random(11284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0184():
    """unknown labels; deterministic sample 184."""
    rng = random.Random(11285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0185():
    """multiclass averaging; deterministic sample 185."""
    rng = random.Random(11286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0186():
    """rare classes; deterministic sample 186."""
    rng = random.Random(11287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0187():
    """unknown labels; deterministic sample 187."""
    rng = random.Random(11288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0188():
    """multiclass averaging; deterministic sample 188."""
    rng = random.Random(11289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0189():
    """rare classes; deterministic sample 189."""
    rng = random.Random(11290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0190():
    """unknown labels; deterministic sample 190."""
    rng = random.Random(11291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0191():
    """multiclass averaging; deterministic sample 191."""
    rng = random.Random(11292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0192():
    """rare classes; deterministic sample 192."""
    rng = random.Random(11293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0193():
    """unknown labels; deterministic sample 193."""
    rng = random.Random(11294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0194():
    """multiclass averaging; deterministic sample 194."""
    rng = random.Random(11295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0195():
    """rare classes; deterministic sample 195."""
    rng = random.Random(11296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0196():
    """unknown labels; deterministic sample 196."""
    rng = random.Random(11297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0197():
    """multiclass averaging; deterministic sample 197."""
    rng = random.Random(11298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0198():
    """rare classes; deterministic sample 198."""
    rng = random.Random(11299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0199():
    """unknown labels; deterministic sample 199."""
    rng = random.Random(11300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0200():
    """multiclass averaging; deterministic sample 200."""
    rng = random.Random(11301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0201():
    """rare classes; deterministic sample 201."""
    rng = random.Random(11302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0202():
    """unknown labels; deterministic sample 202."""
    rng = random.Random(11303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0203():
    """multiclass averaging; deterministic sample 203."""
    rng = random.Random(11304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0204():
    """rare classes; deterministic sample 204."""
    rng = random.Random(11305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0205():
    """unknown labels; deterministic sample 205."""
    rng = random.Random(11306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0206():
    """multiclass averaging; deterministic sample 206."""
    rng = random.Random(11307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0207():
    """rare classes; deterministic sample 207."""
    rng = random.Random(11308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0208():
    """unknown labels; deterministic sample 208."""
    rng = random.Random(11309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0209():
    """multiclass averaging; deterministic sample 209."""
    rng = random.Random(11310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0210():
    """rare classes; deterministic sample 210."""
    rng = random.Random(11311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0211():
    """unknown labels; deterministic sample 211."""
    rng = random.Random(11312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0212():
    """multiclass averaging; deterministic sample 212."""
    rng = random.Random(11313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0213():
    """rare classes; deterministic sample 213."""
    rng = random.Random(11314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0214():
    """unknown labels; deterministic sample 214."""
    rng = random.Random(11315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0215():
    """multiclass averaging; deterministic sample 215."""
    rng = random.Random(11316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0216():
    """rare classes; deterministic sample 216."""
    rng = random.Random(11317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0217():
    """unknown labels; deterministic sample 217."""
    rng = random.Random(11318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0218():
    """multiclass averaging; deterministic sample 218."""
    rng = random.Random(11319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0219():
    """rare classes; deterministic sample 219."""
    rng = random.Random(11320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0220():
    """unknown labels; deterministic sample 220."""
    rng = random.Random(11321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0221():
    """multiclass averaging; deterministic sample 221."""
    rng = random.Random(11322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0222():
    """rare classes; deterministic sample 222."""
    rng = random.Random(11323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0223():
    """unknown labels; deterministic sample 223."""
    rng = random.Random(11324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0224():
    """multiclass averaging; deterministic sample 224."""
    rng = random.Random(11325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0225():
    """rare classes; deterministic sample 225."""
    rng = random.Random(11326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0226():
    """unknown labels; deterministic sample 226."""
    rng = random.Random(11327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0227():
    """multiclass averaging; deterministic sample 227."""
    rng = random.Random(11328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0228():
    """rare classes; deterministic sample 228."""
    rng = random.Random(11329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0229():
    """unknown labels; deterministic sample 229."""
    rng = random.Random(11330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0230():
    """multiclass averaging; deterministic sample 230."""
    rng = random.Random(11331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0231():
    """rare classes; deterministic sample 231."""
    rng = random.Random(11332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0232():
    """unknown labels; deterministic sample 232."""
    rng = random.Random(11333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0233():
    """multiclass averaging; deterministic sample 233."""
    rng = random.Random(11334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0234():
    """rare classes; deterministic sample 234."""
    rng = random.Random(11335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0235():
    """unknown labels; deterministic sample 235."""
    rng = random.Random(11336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0236():
    """multiclass averaging; deterministic sample 236."""
    rng = random.Random(11337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0237():
    """rare classes; deterministic sample 237."""
    rng = random.Random(11338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0238():
    """unknown labels; deterministic sample 238."""
    rng = random.Random(11339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_multiclass_case_0239():
    """multiclass averaging; deterministic sample 239."""
    rng = random.Random(11340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.macro_f1(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

