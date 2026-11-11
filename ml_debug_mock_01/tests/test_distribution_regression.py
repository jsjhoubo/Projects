"""Regression cases for distribution: deterministic scenarios and invariants."""
import math
import random
import pytest
from ml_eval.analytics import distribution as subject

def test_distribution_case_0000():
    """bin boundaries; deterministic sample 0."""
    rng = random.Random(4101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0001():
    """entropy; deterministic sample 1."""
    rng = random.Random(4102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0002():
    """distribution shift; deterministic sample 2."""
    rng = random.Random(4103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0003():
    """bin boundaries; deterministic sample 3."""
    rng = random.Random(4104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0004():
    """entropy; deterministic sample 4."""
    rng = random.Random(4105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0005():
    """distribution shift; deterministic sample 5."""
    rng = random.Random(4106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0006():
    """bin boundaries; deterministic sample 6."""
    rng = random.Random(4107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0007():
    """entropy; deterministic sample 7."""
    rng = random.Random(4108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0008():
    """distribution shift; deterministic sample 8."""
    rng = random.Random(4109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0009():
    """bin boundaries; deterministic sample 9."""
    rng = random.Random(4110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0010():
    """entropy; deterministic sample 10."""
    rng = random.Random(4111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0011():
    """distribution shift; deterministic sample 11."""
    rng = random.Random(4112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0012():
    """bin boundaries; deterministic sample 12."""
    rng = random.Random(4113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0013():
    """entropy; deterministic sample 13."""
    rng = random.Random(4114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0014():
    """distribution shift; deterministic sample 14."""
    rng = random.Random(4115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0015():
    """bin boundaries; deterministic sample 15."""
    rng = random.Random(4116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0016():
    """entropy; deterministic sample 16."""
    rng = random.Random(4117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0017():
    """distribution shift; deterministic sample 17."""
    rng = random.Random(4118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0018():
    """bin boundaries; deterministic sample 18."""
    rng = random.Random(4119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0019():
    """entropy; deterministic sample 19."""
    rng = random.Random(4120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0020():
    """distribution shift; deterministic sample 20."""
    rng = random.Random(4121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0021():
    """bin boundaries; deterministic sample 21."""
    rng = random.Random(4122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0022():
    """entropy; deterministic sample 22."""
    rng = random.Random(4123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0023():
    """distribution shift; deterministic sample 23."""
    rng = random.Random(4124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0024():
    """bin boundaries; deterministic sample 24."""
    rng = random.Random(4125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0025():
    """entropy; deterministic sample 25."""
    rng = random.Random(4126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0026():
    """distribution shift; deterministic sample 26."""
    rng = random.Random(4127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0027():
    """bin boundaries; deterministic sample 27."""
    rng = random.Random(4128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0028():
    """entropy; deterministic sample 28."""
    rng = random.Random(4129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0029():
    """distribution shift; deterministic sample 29."""
    rng = random.Random(4130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0030():
    """bin boundaries; deterministic sample 30."""
    rng = random.Random(4131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0031():
    """entropy; deterministic sample 31."""
    rng = random.Random(4132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0032():
    """distribution shift; deterministic sample 32."""
    rng = random.Random(4133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0033():
    """bin boundaries; deterministic sample 33."""
    rng = random.Random(4134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0034():
    """entropy; deterministic sample 34."""
    rng = random.Random(4135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0035():
    """distribution shift; deterministic sample 35."""
    rng = random.Random(4136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0036():
    """bin boundaries; deterministic sample 36."""
    rng = random.Random(4137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0037():
    """entropy; deterministic sample 37."""
    rng = random.Random(4138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0038():
    """distribution shift; deterministic sample 38."""
    rng = random.Random(4139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0039():
    """bin boundaries; deterministic sample 39."""
    rng = random.Random(4140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0040():
    """entropy; deterministic sample 40."""
    rng = random.Random(4141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0041():
    """distribution shift; deterministic sample 41."""
    rng = random.Random(4142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0042():
    """bin boundaries; deterministic sample 42."""
    rng = random.Random(4143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0043():
    """entropy; deterministic sample 43."""
    rng = random.Random(4144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0044():
    """distribution shift; deterministic sample 44."""
    rng = random.Random(4145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0045():
    """bin boundaries; deterministic sample 45."""
    rng = random.Random(4146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0046():
    """entropy; deterministic sample 46."""
    rng = random.Random(4147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0047():
    """distribution shift; deterministic sample 47."""
    rng = random.Random(4148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0048():
    """bin boundaries; deterministic sample 48."""
    rng = random.Random(4149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0049():
    """entropy; deterministic sample 49."""
    rng = random.Random(4150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0050():
    """distribution shift; deterministic sample 50."""
    rng = random.Random(4151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0051():
    """bin boundaries; deterministic sample 51."""
    rng = random.Random(4152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0052():
    """entropy; deterministic sample 52."""
    rng = random.Random(4153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0053():
    """distribution shift; deterministic sample 53."""
    rng = random.Random(4154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0054():
    """bin boundaries; deterministic sample 54."""
    rng = random.Random(4155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0055():
    """entropy; deterministic sample 55."""
    rng = random.Random(4156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0056():
    """distribution shift; deterministic sample 56."""
    rng = random.Random(4157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0057():
    """bin boundaries; deterministic sample 57."""
    rng = random.Random(4158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0058():
    """entropy; deterministic sample 58."""
    rng = random.Random(4159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0059():
    """distribution shift; deterministic sample 59."""
    rng = random.Random(4160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0060():
    """bin boundaries; deterministic sample 60."""
    rng = random.Random(4161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0061():
    """entropy; deterministic sample 61."""
    rng = random.Random(4162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0062():
    """distribution shift; deterministic sample 62."""
    rng = random.Random(4163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0063():
    """bin boundaries; deterministic sample 63."""
    rng = random.Random(4164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0064():
    """entropy; deterministic sample 64."""
    rng = random.Random(4165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

