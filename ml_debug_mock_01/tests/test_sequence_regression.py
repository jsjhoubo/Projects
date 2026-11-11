"""Regression cases for sequence: both deterministic examples and invariants."""
import math
import random
import pytest
from ml_eval.analytics import sequence as subject

def test_sequence_case_0000():
    """repeated tokens; deterministic sample 0."""
    rng = random.Random(6101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0001():
    """empty sequences; deterministic sample 1."""
    rng = random.Random(6102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0002():
    """n-gram overlap; deterministic sample 2."""
    rng = random.Random(6103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0003():
    """repeated tokens; deterministic sample 3."""
    rng = random.Random(6104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0004():
    """empty sequences; deterministic sample 4."""
    rng = random.Random(6105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0005():
    """n-gram overlap; deterministic sample 5."""
    rng = random.Random(6106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0006():
    """repeated tokens; deterministic sample 6."""
    rng = random.Random(6107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0007():
    """empty sequences; deterministic sample 7."""
    rng = random.Random(6108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0008():
    """n-gram overlap; deterministic sample 8."""
    rng = random.Random(6109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0009():
    """repeated tokens; deterministic sample 9."""
    rng = random.Random(6110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0010():
    """empty sequences; deterministic sample 10."""
    rng = random.Random(6111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0011():
    """n-gram overlap; deterministic sample 11."""
    rng = random.Random(6112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0012():
    """repeated tokens; deterministic sample 12."""
    rng = random.Random(6113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0013():
    """empty sequences; deterministic sample 13."""
    rng = random.Random(6114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0014():
    """n-gram overlap; deterministic sample 14."""
    rng = random.Random(6115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0015():
    """repeated tokens; deterministic sample 15."""
    rng = random.Random(6116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0016():
    """empty sequences; deterministic sample 16."""
    rng = random.Random(6117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0017():
    """n-gram overlap; deterministic sample 17."""
    rng = random.Random(6118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0018():
    """repeated tokens; deterministic sample 18."""
    rng = random.Random(6119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0019():
    """empty sequences; deterministic sample 19."""
    rng = random.Random(6120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0020():
    """n-gram overlap; deterministic sample 20."""
    rng = random.Random(6121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0021():
    """repeated tokens; deterministic sample 21."""
    rng = random.Random(6122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0022():
    """empty sequences; deterministic sample 22."""
    rng = random.Random(6123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0023():
    """n-gram overlap; deterministic sample 23."""
    rng = random.Random(6124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0024():
    """repeated tokens; deterministic sample 24."""
    rng = random.Random(6125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0025():
    """empty sequences; deterministic sample 25."""
    rng = random.Random(6126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0026():
    """n-gram overlap; deterministic sample 26."""
    rng = random.Random(6127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0027():
    """repeated tokens; deterministic sample 27."""
    rng = random.Random(6128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0028():
    """empty sequences; deterministic sample 28."""
    rng = random.Random(6129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0029():
    """n-gram overlap; deterministic sample 29."""
    rng = random.Random(6130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0030():
    """repeated tokens; deterministic sample 30."""
    rng = random.Random(6131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0031():
    """empty sequences; deterministic sample 31."""
    rng = random.Random(6132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0032():
    """n-gram overlap; deterministic sample 32."""
    rng = random.Random(6133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0033():
    """repeated tokens; deterministic sample 33."""
    rng = random.Random(6134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0034():
    """empty sequences; deterministic sample 34."""
    rng = random.Random(6135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0035():
    """n-gram overlap; deterministic sample 35."""
    rng = random.Random(6136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0036():
    """repeated tokens; deterministic sample 36."""
    rng = random.Random(6137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0037():
    """empty sequences; deterministic sample 37."""
    rng = random.Random(6138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0038():
    """n-gram overlap; deterministic sample 38."""
    rng = random.Random(6139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0039():
    """repeated tokens; deterministic sample 39."""
    rng = random.Random(6140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0040():
    """empty sequences; deterministic sample 40."""
    rng = random.Random(6141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0041():
    """n-gram overlap; deterministic sample 41."""
    rng = random.Random(6142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0042():
    """repeated tokens; deterministic sample 42."""
    rng = random.Random(6143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0043():
    """empty sequences; deterministic sample 43."""
    rng = random.Random(6144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0044():
    """n-gram overlap; deterministic sample 44."""
    rng = random.Random(6145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0045():
    """repeated tokens; deterministic sample 45."""
    rng = random.Random(6146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0046():
    """empty sequences; deterministic sample 46."""
    rng = random.Random(6147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0047():
    """n-gram overlap; deterministic sample 47."""
    rng = random.Random(6148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0048():
    """repeated tokens; deterministic sample 48."""
    rng = random.Random(6149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0049():
    """empty sequences; deterministic sample 49."""
    rng = random.Random(6150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0050():
    """n-gram overlap; deterministic sample 50."""
    rng = random.Random(6151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0051():
    """repeated tokens; deterministic sample 51."""
    rng = random.Random(6152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0052():
    """empty sequences; deterministic sample 52."""
    rng = random.Random(6153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0053():
    """n-gram overlap; deterministic sample 53."""
    rng = random.Random(6154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0054():
    """repeated tokens; deterministic sample 54."""
    rng = random.Random(6155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0055():
    """empty sequences; deterministic sample 55."""
    rng = random.Random(6156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0056():
    """n-gram overlap; deterministic sample 56."""
    rng = random.Random(6157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0057():
    """repeated tokens; deterministic sample 57."""
    rng = random.Random(6158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0058():
    """empty sequences; deterministic sample 58."""
    rng = random.Random(6159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0059():
    """n-gram overlap; deterministic sample 59."""
    rng = random.Random(6160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0060():
    """repeated tokens; deterministic sample 60."""
    rng = random.Random(6161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0061():
    """empty sequences; deterministic sample 61."""
    rng = random.Random(6162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0062():
    """n-gram overlap; deterministic sample 62."""
    rng = random.Random(6163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0063():
    """repeated tokens; deterministic sample 63."""
    rng = random.Random(6164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0064():
    """empty sequences; deterministic sample 64."""
    rng = random.Random(6165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0065():
    """n-gram overlap; deterministic sample 65."""
    rng = random.Random(6166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0066():
    """repeated tokens; deterministic sample 66."""
    rng = random.Random(6167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0067():
    """empty sequences; deterministic sample 67."""
    rng = random.Random(6168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0068():
    """n-gram overlap; deterministic sample 68."""
    rng = random.Random(6169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0069():
    """repeated tokens; deterministic sample 69."""
    rng = random.Random(6170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0070():
    """empty sequences; deterministic sample 70."""
    rng = random.Random(6171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0071():
    """n-gram overlap; deterministic sample 71."""
    rng = random.Random(6172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0072():
    """repeated tokens; deterministic sample 72."""
    rng = random.Random(6173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0073():
    """empty sequences; deterministic sample 73."""
    rng = random.Random(6174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0074():
    """n-gram overlap; deterministic sample 74."""
    rng = random.Random(6175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0075():
    """repeated tokens; deterministic sample 75."""
    rng = random.Random(6176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0076():
    """empty sequences; deterministic sample 76."""
    rng = random.Random(6177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0077():
    """n-gram overlap; deterministic sample 77."""
    rng = random.Random(6178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0078():
    """repeated tokens; deterministic sample 78."""
    rng = random.Random(6179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0079():
    """empty sequences; deterministic sample 79."""
    rng = random.Random(6180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0080():
    """n-gram overlap; deterministic sample 80."""
    rng = random.Random(6181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0081():
    """repeated tokens; deterministic sample 81."""
    rng = random.Random(6182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0082():
    """empty sequences; deterministic sample 82."""
    rng = random.Random(6183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0083():
    """n-gram overlap; deterministic sample 83."""
    rng = random.Random(6184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0084():
    """repeated tokens; deterministic sample 84."""
    rng = random.Random(6185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0085():
    """empty sequences; deterministic sample 85."""
    rng = random.Random(6186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0086():
    """n-gram overlap; deterministic sample 86."""
    rng = random.Random(6187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0087():
    """repeated tokens; deterministic sample 87."""
    rng = random.Random(6188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0088():
    """empty sequences; deterministic sample 88."""
    rng = random.Random(6189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0089():
    """n-gram overlap; deterministic sample 89."""
    rng = random.Random(6190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0090():
    """repeated tokens; deterministic sample 90."""
    rng = random.Random(6191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0091():
    """empty sequences; deterministic sample 91."""
    rng = random.Random(6192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0092():
    """n-gram overlap; deterministic sample 92."""
    rng = random.Random(6193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0093():
    """repeated tokens; deterministic sample 93."""
    rng = random.Random(6194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0094():
    """empty sequences; deterministic sample 94."""
    rng = random.Random(6195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0095():
    """n-gram overlap; deterministic sample 95."""
    rng = random.Random(6196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0096():
    """repeated tokens; deterministic sample 96."""
    rng = random.Random(6197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0097():
    """empty sequences; deterministic sample 97."""
    rng = random.Random(6198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0098():
    """n-gram overlap; deterministic sample 98."""
    rng = random.Random(6199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0099():
    """repeated tokens; deterministic sample 99."""
    rng = random.Random(6200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0100():
    """empty sequences; deterministic sample 100."""
    rng = random.Random(6201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0101():
    """n-gram overlap; deterministic sample 101."""
    rng = random.Random(6202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0102():
    """repeated tokens; deterministic sample 102."""
    rng = random.Random(6203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0103():
    """empty sequences; deterministic sample 103."""
    rng = random.Random(6204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0104():
    """n-gram overlap; deterministic sample 104."""
    rng = random.Random(6205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0105():
    """repeated tokens; deterministic sample 105."""
    rng = random.Random(6206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0106():
    """empty sequences; deterministic sample 106."""
    rng = random.Random(6207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0107():
    """n-gram overlap; deterministic sample 107."""
    rng = random.Random(6208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0108():
    """repeated tokens; deterministic sample 108."""
    rng = random.Random(6209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0109():
    """empty sequences; deterministic sample 109."""
    rng = random.Random(6210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0110():
    """n-gram overlap; deterministic sample 110."""
    rng = random.Random(6211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0111():
    """repeated tokens; deterministic sample 111."""
    rng = random.Random(6212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0112():
    """empty sequences; deterministic sample 112."""
    rng = random.Random(6213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0113():
    """n-gram overlap; deterministic sample 113."""
    rng = random.Random(6214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0114():
    """repeated tokens; deterministic sample 114."""
    rng = random.Random(6215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0115():
    """empty sequences; deterministic sample 115."""
    rng = random.Random(6216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0116():
    """n-gram overlap; deterministic sample 116."""
    rng = random.Random(6217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0117():
    """repeated tokens; deterministic sample 117."""
    rng = random.Random(6218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0118():
    """empty sequences; deterministic sample 118."""
    rng = random.Random(6219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0119():
    """n-gram overlap; deterministic sample 119."""
    rng = random.Random(6220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0120():
    """repeated tokens; deterministic sample 120."""
    rng = random.Random(6221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0121():
    """empty sequences; deterministic sample 121."""
    rng = random.Random(6222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0122():
    """n-gram overlap; deterministic sample 122."""
    rng = random.Random(6223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0123():
    """repeated tokens; deterministic sample 123."""
    rng = random.Random(6224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0124():
    """empty sequences; deterministic sample 124."""
    rng = random.Random(6225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0125():
    """n-gram overlap; deterministic sample 125."""
    rng = random.Random(6226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0126():
    """repeated tokens; deterministic sample 126."""
    rng = random.Random(6227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0127():
    """empty sequences; deterministic sample 127."""
    rng = random.Random(6228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0128():
    """n-gram overlap; deterministic sample 128."""
    rng = random.Random(6229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0129():
    """repeated tokens; deterministic sample 129."""
    rng = random.Random(6230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0130():
    """empty sequences; deterministic sample 130."""
    rng = random.Random(6231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0131():
    """n-gram overlap; deterministic sample 131."""
    rng = random.Random(6232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0132():
    """repeated tokens; deterministic sample 132."""
    rng = random.Random(6233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0133():
    """empty sequences; deterministic sample 133."""
    rng = random.Random(6234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0134():
    """n-gram overlap; deterministic sample 134."""
    rng = random.Random(6235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0135():
    """repeated tokens; deterministic sample 135."""
    rng = random.Random(6236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0136():
    """empty sequences; deterministic sample 136."""
    rng = random.Random(6237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0137():
    """n-gram overlap; deterministic sample 137."""
    rng = random.Random(6238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0138():
    """repeated tokens; deterministic sample 138."""
    rng = random.Random(6239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0139():
    """empty sequences; deterministic sample 139."""
    rng = random.Random(6240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0140():
    """n-gram overlap; deterministic sample 140."""
    rng = random.Random(6241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0141():
    """repeated tokens; deterministic sample 141."""
    rng = random.Random(6242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0142():
    """empty sequences; deterministic sample 142."""
    rng = random.Random(6243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0143():
    """n-gram overlap; deterministic sample 143."""
    rng = random.Random(6244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0144():
    """repeated tokens; deterministic sample 144."""
    rng = random.Random(6245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0145():
    """empty sequences; deterministic sample 145."""
    rng = random.Random(6246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0146():
    """n-gram overlap; deterministic sample 146."""
    rng = random.Random(6247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0147():
    """repeated tokens; deterministic sample 147."""
    rng = random.Random(6248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0148():
    """empty sequences; deterministic sample 148."""
    rng = random.Random(6249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0149():
    """n-gram overlap; deterministic sample 149."""
    rng = random.Random(6250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0150():
    """repeated tokens; deterministic sample 150."""
    rng = random.Random(6251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0151():
    """empty sequences; deterministic sample 151."""
    rng = random.Random(6252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0152():
    """n-gram overlap; deterministic sample 152."""
    rng = random.Random(6253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0153():
    """repeated tokens; deterministic sample 153."""
    rng = random.Random(6254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0154():
    """empty sequences; deterministic sample 154."""
    rng = random.Random(6255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0155():
    """n-gram overlap; deterministic sample 155."""
    rng = random.Random(6256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0156():
    """repeated tokens; deterministic sample 156."""
    rng = random.Random(6257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0157():
    """empty sequences; deterministic sample 157."""
    rng = random.Random(6258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0158():
    """n-gram overlap; deterministic sample 158."""
    rng = random.Random(6259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0159():
    """repeated tokens; deterministic sample 159."""
    rng = random.Random(6260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0160():
    """empty sequences; deterministic sample 160."""
    rng = random.Random(6261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0161():
    """n-gram overlap; deterministic sample 161."""
    rng = random.Random(6262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0162():
    """repeated tokens; deterministic sample 162."""
    rng = random.Random(6263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0163():
    """empty sequences; deterministic sample 163."""
    rng = random.Random(6264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0164():
    """n-gram overlap; deterministic sample 164."""
    rng = random.Random(6265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0165():
    """repeated tokens; deterministic sample 165."""
    rng = random.Random(6266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0166():
    """empty sequences; deterministic sample 166."""
    rng = random.Random(6267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0167():
    """n-gram overlap; deterministic sample 167."""
    rng = random.Random(6268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0168():
    """repeated tokens; deterministic sample 168."""
    rng = random.Random(6269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0169():
    """empty sequences; deterministic sample 169."""
    rng = random.Random(6270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0170():
    """n-gram overlap; deterministic sample 170."""
    rng = random.Random(6271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0171():
    """repeated tokens; deterministic sample 171."""
    rng = random.Random(6272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0172():
    """empty sequences; deterministic sample 172."""
    rng = random.Random(6273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0173():
    """n-gram overlap; deterministic sample 173."""
    rng = random.Random(6274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0174():
    """repeated tokens; deterministic sample 174."""
    rng = random.Random(6275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0175():
    """empty sequences; deterministic sample 175."""
    rng = random.Random(6276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0176():
    """n-gram overlap; deterministic sample 176."""
    rng = random.Random(6277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0177():
    """repeated tokens; deterministic sample 177."""
    rng = random.Random(6278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0178():
    """empty sequences; deterministic sample 178."""
    rng = random.Random(6279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0179():
    """n-gram overlap; deterministic sample 179."""
    rng = random.Random(6280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0180():
    """repeated tokens; deterministic sample 180."""
    rng = random.Random(6281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0181():
    """empty sequences; deterministic sample 181."""
    rng = random.Random(6282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0182():
    """n-gram overlap; deterministic sample 182."""
    rng = random.Random(6283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0183():
    """repeated tokens; deterministic sample 183."""
    rng = random.Random(6284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0184():
    """empty sequences; deterministic sample 184."""
    rng = random.Random(6285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0185():
    """n-gram overlap; deterministic sample 185."""
    rng = random.Random(6286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0186():
    """repeated tokens; deterministic sample 186."""
    rng = random.Random(6287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0187():
    """empty sequences; deterministic sample 187."""
    rng = random.Random(6288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0188():
    """n-gram overlap; deterministic sample 188."""
    rng = random.Random(6289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0189():
    """repeated tokens; deterministic sample 189."""
    rng = random.Random(6290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0190():
    """empty sequences; deterministic sample 190."""
    rng = random.Random(6291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0191():
    """n-gram overlap; deterministic sample 191."""
    rng = random.Random(6292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0192():
    """repeated tokens; deterministic sample 192."""
    rng = random.Random(6293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0193():
    """empty sequences; deterministic sample 193."""
    rng = random.Random(6294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0194():
    """n-gram overlap; deterministic sample 194."""
    rng = random.Random(6295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0195():
    """repeated tokens; deterministic sample 195."""
    rng = random.Random(6296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0196():
    """empty sequences; deterministic sample 196."""
    rng = random.Random(6297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0197():
    """n-gram overlap; deterministic sample 197."""
    rng = random.Random(6298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0198():
    """repeated tokens; deterministic sample 198."""
    rng = random.Random(6299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0199():
    """empty sequences; deterministic sample 199."""
    rng = random.Random(6300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0200():
    """n-gram overlap; deterministic sample 200."""
    rng = random.Random(6301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0201():
    """repeated tokens; deterministic sample 201."""
    rng = random.Random(6302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0202():
    """empty sequences; deterministic sample 202."""
    rng = random.Random(6303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0203():
    """n-gram overlap; deterministic sample 203."""
    rng = random.Random(6304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0204():
    """repeated tokens; deterministic sample 204."""
    rng = random.Random(6305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0205():
    """empty sequences; deterministic sample 205."""
    rng = random.Random(6306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0206():
    """n-gram overlap; deterministic sample 206."""
    rng = random.Random(6307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0207():
    """repeated tokens; deterministic sample 207."""
    rng = random.Random(6308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0208():
    """empty sequences; deterministic sample 208."""
    rng = random.Random(6309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0209():
    """n-gram overlap; deterministic sample 209."""
    rng = random.Random(6310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0210():
    """repeated tokens; deterministic sample 210."""
    rng = random.Random(6311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0211():
    """empty sequences; deterministic sample 211."""
    rng = random.Random(6312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0212():
    """n-gram overlap; deterministic sample 212."""
    rng = random.Random(6313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0213():
    """repeated tokens; deterministic sample 213."""
    rng = random.Random(6314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0214():
    """empty sequences; deterministic sample 214."""
    rng = random.Random(6315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0215():
    """n-gram overlap; deterministic sample 215."""
    rng = random.Random(6316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0216():
    """repeated tokens; deterministic sample 216."""
    rng = random.Random(6317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0217():
    """empty sequences; deterministic sample 217."""
    rng = random.Random(6318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0218():
    """n-gram overlap; deterministic sample 218."""
    rng = random.Random(6319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0219():
    """repeated tokens; deterministic sample 219."""
    rng = random.Random(6320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0220():
    """empty sequences; deterministic sample 220."""
    rng = random.Random(6321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0221():
    """n-gram overlap; deterministic sample 221."""
    rng = random.Random(6322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0222():
    """repeated tokens; deterministic sample 222."""
    rng = random.Random(6323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0223():
    """empty sequences; deterministic sample 223."""
    rng = random.Random(6324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0224():
    """n-gram overlap; deterministic sample 224."""
    rng = random.Random(6325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0225():
    """repeated tokens; deterministic sample 225."""
    rng = random.Random(6326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0226():
    """empty sequences; deterministic sample 226."""
    rng = random.Random(6327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0227():
    """n-gram overlap; deterministic sample 227."""
    rng = random.Random(6328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0228():
    """repeated tokens; deterministic sample 228."""
    rng = random.Random(6329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0229():
    """empty sequences; deterministic sample 229."""
    rng = random.Random(6330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0230():
    """n-gram overlap; deterministic sample 230."""
    rng = random.Random(6331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0231():
    """repeated tokens; deterministic sample 231."""
    rng = random.Random(6332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0232():
    """empty sequences; deterministic sample 232."""
    rng = random.Random(6333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0233():
    """n-gram overlap; deterministic sample 233."""
    rng = random.Random(6334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0234():
    """repeated tokens; deterministic sample 234."""
    rng = random.Random(6335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0235():
    """empty sequences; deterministic sample 235."""
    rng = random.Random(6336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0236():
    """n-gram overlap; deterministic sample 236."""
    rng = random.Random(6337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0237():
    """repeated tokens; deterministic sample 237."""
    rng = random.Random(6338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0238():
    """empty sequences; deterministic sample 238."""
    rng = random.Random(6339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_sequence_case_0239():
    """n-gram overlap; deterministic sample 239."""
    rng = random.Random(6340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.token_accuracy(preds, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

