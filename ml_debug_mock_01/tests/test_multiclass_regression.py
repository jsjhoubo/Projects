"""Regression cases for multiclass: deterministic scenarios and invariants."""
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

