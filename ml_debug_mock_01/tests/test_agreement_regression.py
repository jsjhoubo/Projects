"""Regression cases for agreement: both deterministic examples and invariants."""
import math
import random
import pytest
from ml_eval.analytics import agreement as subject

def test_agreement_case_0000():
    """chance agreement; deterministic sample 0."""
    rng = random.Random(10101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0001():
    """imbalanced votes; deterministic sample 1."""
    rng = random.Random(10102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0002():
    """annotator bias; deterministic sample 2."""
    rng = random.Random(10103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0003():
    """chance agreement; deterministic sample 3."""
    rng = random.Random(10104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0004():
    """imbalanced votes; deterministic sample 4."""
    rng = random.Random(10105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0005():
    """annotator bias; deterministic sample 5."""
    rng = random.Random(10106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0006():
    """chance agreement; deterministic sample 6."""
    rng = random.Random(10107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0007():
    """imbalanced votes; deterministic sample 7."""
    rng = random.Random(10108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0008():
    """annotator bias; deterministic sample 8."""
    rng = random.Random(10109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0009():
    """chance agreement; deterministic sample 9."""
    rng = random.Random(10110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0010():
    """imbalanced votes; deterministic sample 10."""
    rng = random.Random(10111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0011():
    """annotator bias; deterministic sample 11."""
    rng = random.Random(10112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0012():
    """chance agreement; deterministic sample 12."""
    rng = random.Random(10113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0013():
    """imbalanced votes; deterministic sample 13."""
    rng = random.Random(10114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0014():
    """annotator bias; deterministic sample 14."""
    rng = random.Random(10115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0015():
    """chance agreement; deterministic sample 15."""
    rng = random.Random(10116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0016():
    """imbalanced votes; deterministic sample 16."""
    rng = random.Random(10117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0017():
    """annotator bias; deterministic sample 17."""
    rng = random.Random(10118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0018():
    """chance agreement; deterministic sample 18."""
    rng = random.Random(10119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0019():
    """imbalanced votes; deterministic sample 19."""
    rng = random.Random(10120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0020():
    """annotator bias; deterministic sample 20."""
    rng = random.Random(10121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0021():
    """chance agreement; deterministic sample 21."""
    rng = random.Random(10122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0022():
    """imbalanced votes; deterministic sample 22."""
    rng = random.Random(10123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0023():
    """annotator bias; deterministic sample 23."""
    rng = random.Random(10124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0024():
    """chance agreement; deterministic sample 24."""
    rng = random.Random(10125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0025():
    """imbalanced votes; deterministic sample 25."""
    rng = random.Random(10126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0026():
    """annotator bias; deterministic sample 26."""
    rng = random.Random(10127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0027():
    """chance agreement; deterministic sample 27."""
    rng = random.Random(10128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0028():
    """imbalanced votes; deterministic sample 28."""
    rng = random.Random(10129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0029():
    """annotator bias; deterministic sample 29."""
    rng = random.Random(10130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0030():
    """chance agreement; deterministic sample 30."""
    rng = random.Random(10131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0031():
    """imbalanced votes; deterministic sample 31."""
    rng = random.Random(10132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0032():
    """annotator bias; deterministic sample 32."""
    rng = random.Random(10133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0033():
    """chance agreement; deterministic sample 33."""
    rng = random.Random(10134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0034():
    """imbalanced votes; deterministic sample 34."""
    rng = random.Random(10135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0035():
    """annotator bias; deterministic sample 35."""
    rng = random.Random(10136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0036():
    """chance agreement; deterministic sample 36."""
    rng = random.Random(10137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0037():
    """imbalanced votes; deterministic sample 37."""
    rng = random.Random(10138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0038():
    """annotator bias; deterministic sample 38."""
    rng = random.Random(10139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0039():
    """chance agreement; deterministic sample 39."""
    rng = random.Random(10140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0040():
    """imbalanced votes; deterministic sample 40."""
    rng = random.Random(10141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0041():
    """annotator bias; deterministic sample 41."""
    rng = random.Random(10142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0042():
    """chance agreement; deterministic sample 42."""
    rng = random.Random(10143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0043():
    """imbalanced votes; deterministic sample 43."""
    rng = random.Random(10144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0044():
    """annotator bias; deterministic sample 44."""
    rng = random.Random(10145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0045():
    """chance agreement; deterministic sample 45."""
    rng = random.Random(10146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0046():
    """imbalanced votes; deterministic sample 46."""
    rng = random.Random(10147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0047():
    """annotator bias; deterministic sample 47."""
    rng = random.Random(10148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0048():
    """chance agreement; deterministic sample 48."""
    rng = random.Random(10149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0049():
    """imbalanced votes; deterministic sample 49."""
    rng = random.Random(10150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0050():
    """annotator bias; deterministic sample 50."""
    rng = random.Random(10151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0051():
    """chance agreement; deterministic sample 51."""
    rng = random.Random(10152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0052():
    """imbalanced votes; deterministic sample 52."""
    rng = random.Random(10153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0053():
    """annotator bias; deterministic sample 53."""
    rng = random.Random(10154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0054():
    """chance agreement; deterministic sample 54."""
    rng = random.Random(10155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0055():
    """imbalanced votes; deterministic sample 55."""
    rng = random.Random(10156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0056():
    """annotator bias; deterministic sample 56."""
    rng = random.Random(10157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0057():
    """chance agreement; deterministic sample 57."""
    rng = random.Random(10158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0058():
    """imbalanced votes; deterministic sample 58."""
    rng = random.Random(10159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0059():
    """annotator bias; deterministic sample 59."""
    rng = random.Random(10160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0060():
    """chance agreement; deterministic sample 60."""
    rng = random.Random(10161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0061():
    """imbalanced votes; deterministic sample 61."""
    rng = random.Random(10162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0062():
    """annotator bias; deterministic sample 62."""
    rng = random.Random(10163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0063():
    """chance agreement; deterministic sample 63."""
    rng = random.Random(10164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0064():
    """imbalanced votes; deterministic sample 64."""
    rng = random.Random(10165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0065():
    """annotator bias; deterministic sample 65."""
    rng = random.Random(10166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0066():
    """chance agreement; deterministic sample 66."""
    rng = random.Random(10167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0067():
    """imbalanced votes; deterministic sample 67."""
    rng = random.Random(10168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0068():
    """annotator bias; deterministic sample 68."""
    rng = random.Random(10169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0069():
    """chance agreement; deterministic sample 69."""
    rng = random.Random(10170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0070():
    """imbalanced votes; deterministic sample 70."""
    rng = random.Random(10171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0071():
    """annotator bias; deterministic sample 71."""
    rng = random.Random(10172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0072():
    """chance agreement; deterministic sample 72."""
    rng = random.Random(10173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0073():
    """imbalanced votes; deterministic sample 73."""
    rng = random.Random(10174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0074():
    """annotator bias; deterministic sample 74."""
    rng = random.Random(10175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0075():
    """chance agreement; deterministic sample 75."""
    rng = random.Random(10176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0076():
    """imbalanced votes; deterministic sample 76."""
    rng = random.Random(10177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0077():
    """annotator bias; deterministic sample 77."""
    rng = random.Random(10178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0078():
    """chance agreement; deterministic sample 78."""
    rng = random.Random(10179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0079():
    """imbalanced votes; deterministic sample 79."""
    rng = random.Random(10180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0080():
    """annotator bias; deterministic sample 80."""
    rng = random.Random(10181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0081():
    """chance agreement; deterministic sample 81."""
    rng = random.Random(10182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0082():
    """imbalanced votes; deterministic sample 82."""
    rng = random.Random(10183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0083():
    """annotator bias; deterministic sample 83."""
    rng = random.Random(10184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0084():
    """chance agreement; deterministic sample 84."""
    rng = random.Random(10185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0085():
    """imbalanced votes; deterministic sample 85."""
    rng = random.Random(10186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0086():
    """annotator bias; deterministic sample 86."""
    rng = random.Random(10187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0087():
    """chance agreement; deterministic sample 87."""
    rng = random.Random(10188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0088():
    """imbalanced votes; deterministic sample 88."""
    rng = random.Random(10189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0089():
    """annotator bias; deterministic sample 89."""
    rng = random.Random(10190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0090():
    """chance agreement; deterministic sample 90."""
    rng = random.Random(10191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0091():
    """imbalanced votes; deterministic sample 91."""
    rng = random.Random(10192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0092():
    """annotator bias; deterministic sample 92."""
    rng = random.Random(10193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0093():
    """chance agreement; deterministic sample 93."""
    rng = random.Random(10194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0094():
    """imbalanced votes; deterministic sample 94."""
    rng = random.Random(10195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0095():
    """annotator bias; deterministic sample 95."""
    rng = random.Random(10196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0096():
    """chance agreement; deterministic sample 96."""
    rng = random.Random(10197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0097():
    """imbalanced votes; deterministic sample 97."""
    rng = random.Random(10198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0098():
    """annotator bias; deterministic sample 98."""
    rng = random.Random(10199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0099():
    """chance agreement; deterministic sample 99."""
    rng = random.Random(10200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0100():
    """imbalanced votes; deterministic sample 100."""
    rng = random.Random(10201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0101():
    """annotator bias; deterministic sample 101."""
    rng = random.Random(10202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0102():
    """chance agreement; deterministic sample 102."""
    rng = random.Random(10203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0103():
    """imbalanced votes; deterministic sample 103."""
    rng = random.Random(10204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0104():
    """annotator bias; deterministic sample 104."""
    rng = random.Random(10205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0105():
    """chance agreement; deterministic sample 105."""
    rng = random.Random(10206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0106():
    """imbalanced votes; deterministic sample 106."""
    rng = random.Random(10207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0107():
    """annotator bias; deterministic sample 107."""
    rng = random.Random(10208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0108():
    """chance agreement; deterministic sample 108."""
    rng = random.Random(10209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0109():
    """imbalanced votes; deterministic sample 109."""
    rng = random.Random(10210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0110():
    """annotator bias; deterministic sample 110."""
    rng = random.Random(10211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0111():
    """chance agreement; deterministic sample 111."""
    rng = random.Random(10212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0112():
    """imbalanced votes; deterministic sample 112."""
    rng = random.Random(10213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0113():
    """annotator bias; deterministic sample 113."""
    rng = random.Random(10214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0114():
    """chance agreement; deterministic sample 114."""
    rng = random.Random(10215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0115():
    """imbalanced votes; deterministic sample 115."""
    rng = random.Random(10216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0116():
    """annotator bias; deterministic sample 116."""
    rng = random.Random(10217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0117():
    """chance agreement; deterministic sample 117."""
    rng = random.Random(10218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0118():
    """imbalanced votes; deterministic sample 118."""
    rng = random.Random(10219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0119():
    """annotator bias; deterministic sample 119."""
    rng = random.Random(10220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0120():
    """chance agreement; deterministic sample 120."""
    rng = random.Random(10221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0121():
    """imbalanced votes; deterministic sample 121."""
    rng = random.Random(10222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0122():
    """annotator bias; deterministic sample 122."""
    rng = random.Random(10223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0123():
    """chance agreement; deterministic sample 123."""
    rng = random.Random(10224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0124():
    """imbalanced votes; deterministic sample 124."""
    rng = random.Random(10225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0125():
    """annotator bias; deterministic sample 125."""
    rng = random.Random(10226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0126():
    """chance agreement; deterministic sample 126."""
    rng = random.Random(10227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0127():
    """imbalanced votes; deterministic sample 127."""
    rng = random.Random(10228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0128():
    """annotator bias; deterministic sample 128."""
    rng = random.Random(10229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0129():
    """chance agreement; deterministic sample 129."""
    rng = random.Random(10230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0130():
    """imbalanced votes; deterministic sample 130."""
    rng = random.Random(10231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0131():
    """annotator bias; deterministic sample 131."""
    rng = random.Random(10232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0132():
    """chance agreement; deterministic sample 132."""
    rng = random.Random(10233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0133():
    """imbalanced votes; deterministic sample 133."""
    rng = random.Random(10234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0134():
    """annotator bias; deterministic sample 134."""
    rng = random.Random(10235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0135():
    """chance agreement; deterministic sample 135."""
    rng = random.Random(10236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0136():
    """imbalanced votes; deterministic sample 136."""
    rng = random.Random(10237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0137():
    """annotator bias; deterministic sample 137."""
    rng = random.Random(10238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0138():
    """chance agreement; deterministic sample 138."""
    rng = random.Random(10239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0139():
    """imbalanced votes; deterministic sample 139."""
    rng = random.Random(10240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0140():
    """annotator bias; deterministic sample 140."""
    rng = random.Random(10241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0141():
    """chance agreement; deterministic sample 141."""
    rng = random.Random(10242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0142():
    """imbalanced votes; deterministic sample 142."""
    rng = random.Random(10243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0143():
    """annotator bias; deterministic sample 143."""
    rng = random.Random(10244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0144():
    """chance agreement; deterministic sample 144."""
    rng = random.Random(10245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0145():
    """imbalanced votes; deterministic sample 145."""
    rng = random.Random(10246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0146():
    """annotator bias; deterministic sample 146."""
    rng = random.Random(10247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0147():
    """chance agreement; deterministic sample 147."""
    rng = random.Random(10248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0148():
    """imbalanced votes; deterministic sample 148."""
    rng = random.Random(10249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0149():
    """annotator bias; deterministic sample 149."""
    rng = random.Random(10250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0150():
    """chance agreement; deterministic sample 150."""
    rng = random.Random(10251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0151():
    """imbalanced votes; deterministic sample 151."""
    rng = random.Random(10252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0152():
    """annotator bias; deterministic sample 152."""
    rng = random.Random(10253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0153():
    """chance agreement; deterministic sample 153."""
    rng = random.Random(10254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0154():
    """imbalanced votes; deterministic sample 154."""
    rng = random.Random(10255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0155():
    """annotator bias; deterministic sample 155."""
    rng = random.Random(10256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0156():
    """chance agreement; deterministic sample 156."""
    rng = random.Random(10257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0157():
    """imbalanced votes; deterministic sample 157."""
    rng = random.Random(10258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0158():
    """annotator bias; deterministic sample 158."""
    rng = random.Random(10259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0159():
    """chance agreement; deterministic sample 159."""
    rng = random.Random(10260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0160():
    """imbalanced votes; deterministic sample 160."""
    rng = random.Random(10261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0161():
    """annotator bias; deterministic sample 161."""
    rng = random.Random(10262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0162():
    """chance agreement; deterministic sample 162."""
    rng = random.Random(10263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0163():
    """imbalanced votes; deterministic sample 163."""
    rng = random.Random(10264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0164():
    """annotator bias; deterministic sample 164."""
    rng = random.Random(10265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0165():
    """chance agreement; deterministic sample 165."""
    rng = random.Random(10266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0166():
    """imbalanced votes; deterministic sample 166."""
    rng = random.Random(10267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0167():
    """annotator bias; deterministic sample 167."""
    rng = random.Random(10268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0168():
    """chance agreement; deterministic sample 168."""
    rng = random.Random(10269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0169():
    """imbalanced votes; deterministic sample 169."""
    rng = random.Random(10270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0170():
    """annotator bias; deterministic sample 170."""
    rng = random.Random(10271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0171():
    """chance agreement; deterministic sample 171."""
    rng = random.Random(10272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0172():
    """imbalanced votes; deterministic sample 172."""
    rng = random.Random(10273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0173():
    """annotator bias; deterministic sample 173."""
    rng = random.Random(10274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0174():
    """chance agreement; deterministic sample 174."""
    rng = random.Random(10275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0175():
    """imbalanced votes; deterministic sample 175."""
    rng = random.Random(10276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0176():
    """annotator bias; deterministic sample 176."""
    rng = random.Random(10277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0177():
    """chance agreement; deterministic sample 177."""
    rng = random.Random(10278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0178():
    """imbalanced votes; deterministic sample 178."""
    rng = random.Random(10279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0179():
    """annotator bias; deterministic sample 179."""
    rng = random.Random(10280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0180():
    """chance agreement; deterministic sample 180."""
    rng = random.Random(10281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0181():
    """imbalanced votes; deterministic sample 181."""
    rng = random.Random(10282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0182():
    """annotator bias; deterministic sample 182."""
    rng = random.Random(10283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0183():
    """chance agreement; deterministic sample 183."""
    rng = random.Random(10284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0184():
    """imbalanced votes; deterministic sample 184."""
    rng = random.Random(10285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0185():
    """annotator bias; deterministic sample 185."""
    rng = random.Random(10286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0186():
    """chance agreement; deterministic sample 186."""
    rng = random.Random(10287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0187():
    """imbalanced votes; deterministic sample 187."""
    rng = random.Random(10288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0188():
    """annotator bias; deterministic sample 188."""
    rng = random.Random(10289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0189():
    """chance agreement; deterministic sample 189."""
    rng = random.Random(10290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0190():
    """imbalanced votes; deterministic sample 190."""
    rng = random.Random(10291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0191():
    """annotator bias; deterministic sample 191."""
    rng = random.Random(10292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0192():
    """chance agreement; deterministic sample 192."""
    rng = random.Random(10293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0193():
    """imbalanced votes; deterministic sample 193."""
    rng = random.Random(10294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0194():
    """annotator bias; deterministic sample 194."""
    rng = random.Random(10295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0195():
    """chance agreement; deterministic sample 195."""
    rng = random.Random(10296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0196():
    """imbalanced votes; deterministic sample 196."""
    rng = random.Random(10297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0197():
    """annotator bias; deterministic sample 197."""
    rng = random.Random(10298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0198():
    """chance agreement; deterministic sample 198."""
    rng = random.Random(10299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0199():
    """imbalanced votes; deterministic sample 199."""
    rng = random.Random(10300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0200():
    """annotator bias; deterministic sample 200."""
    rng = random.Random(10301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0201():
    """chance agreement; deterministic sample 201."""
    rng = random.Random(10302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0202():
    """imbalanced votes; deterministic sample 202."""
    rng = random.Random(10303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0203():
    """annotator bias; deterministic sample 203."""
    rng = random.Random(10304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0204():
    """chance agreement; deterministic sample 204."""
    rng = random.Random(10305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0205():
    """imbalanced votes; deterministic sample 205."""
    rng = random.Random(10306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0206():
    """annotator bias; deterministic sample 206."""
    rng = random.Random(10307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0207():
    """chance agreement; deterministic sample 207."""
    rng = random.Random(10308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0208():
    """imbalanced votes; deterministic sample 208."""
    rng = random.Random(10309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0209():
    """annotator bias; deterministic sample 209."""
    rng = random.Random(10310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0210():
    """chance agreement; deterministic sample 210."""
    rng = random.Random(10311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0211():
    """imbalanced votes; deterministic sample 211."""
    rng = random.Random(10312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0212():
    """annotator bias; deterministic sample 212."""
    rng = random.Random(10313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0213():
    """chance agreement; deterministic sample 213."""
    rng = random.Random(10314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0214():
    """imbalanced votes; deterministic sample 214."""
    rng = random.Random(10315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0215():
    """annotator bias; deterministic sample 215."""
    rng = random.Random(10316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0216():
    """chance agreement; deterministic sample 216."""
    rng = random.Random(10317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0217():
    """imbalanced votes; deterministic sample 217."""
    rng = random.Random(10318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0218():
    """annotator bias; deterministic sample 218."""
    rng = random.Random(10319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0219():
    """chance agreement; deterministic sample 219."""
    rng = random.Random(10320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0220():
    """imbalanced votes; deterministic sample 220."""
    rng = random.Random(10321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0221():
    """annotator bias; deterministic sample 221."""
    rng = random.Random(10322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0222():
    """chance agreement; deterministic sample 222."""
    rng = random.Random(10323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0223():
    """imbalanced votes; deterministic sample 223."""
    rng = random.Random(10324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0224():
    """annotator bias; deterministic sample 224."""
    rng = random.Random(10325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0225():
    """chance agreement; deterministic sample 225."""
    rng = random.Random(10326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0226():
    """imbalanced votes; deterministic sample 226."""
    rng = random.Random(10327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0227():
    """annotator bias; deterministic sample 227."""
    rng = random.Random(10328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0228():
    """chance agreement; deterministic sample 228."""
    rng = random.Random(10329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0229():
    """imbalanced votes; deterministic sample 229."""
    rng = random.Random(10330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0230():
    """annotator bias; deterministic sample 230."""
    rng = random.Random(10331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0231():
    """chance agreement; deterministic sample 231."""
    rng = random.Random(10332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0232():
    """imbalanced votes; deterministic sample 232."""
    rng = random.Random(10333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0233():
    """annotator bias; deterministic sample 233."""
    rng = random.Random(10334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0234():
    """chance agreement; deterministic sample 234."""
    rng = random.Random(10335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0235():
    """imbalanced votes; deterministic sample 235."""
    rng = random.Random(10336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0236():
    """annotator bias; deterministic sample 236."""
    rng = random.Random(10337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0237():
    """chance agreement; deterministic sample 237."""
    rng = random.Random(10338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0238():
    """imbalanced votes; deterministic sample 238."""
    rng = random.Random(10339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_agreement_case_0239():
    """annotator bias; deterministic sample 239."""
    rng = random.Random(10340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.pairwise_agreement(preds,labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

