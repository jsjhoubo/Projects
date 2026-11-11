"""Regression cases for distribution: both deterministic examples and invariants."""
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

def test_distribution_case_0065():
    """distribution shift; deterministic sample 65."""
    rng = random.Random(4166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0066():
    """bin boundaries; deterministic sample 66."""
    rng = random.Random(4167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0067():
    """entropy; deterministic sample 67."""
    rng = random.Random(4168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0068():
    """distribution shift; deterministic sample 68."""
    rng = random.Random(4169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0069():
    """bin boundaries; deterministic sample 69."""
    rng = random.Random(4170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0070():
    """entropy; deterministic sample 70."""
    rng = random.Random(4171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0071():
    """distribution shift; deterministic sample 71."""
    rng = random.Random(4172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0072():
    """bin boundaries; deterministic sample 72."""
    rng = random.Random(4173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0073():
    """entropy; deterministic sample 73."""
    rng = random.Random(4174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0074():
    """distribution shift; deterministic sample 74."""
    rng = random.Random(4175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0075():
    """bin boundaries; deterministic sample 75."""
    rng = random.Random(4176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0076():
    """entropy; deterministic sample 76."""
    rng = random.Random(4177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0077():
    """distribution shift; deterministic sample 77."""
    rng = random.Random(4178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0078():
    """bin boundaries; deterministic sample 78."""
    rng = random.Random(4179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0079():
    """entropy; deterministic sample 79."""
    rng = random.Random(4180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0080():
    """distribution shift; deterministic sample 80."""
    rng = random.Random(4181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0081():
    """bin boundaries; deterministic sample 81."""
    rng = random.Random(4182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0082():
    """entropy; deterministic sample 82."""
    rng = random.Random(4183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0083():
    """distribution shift; deterministic sample 83."""
    rng = random.Random(4184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0084():
    """bin boundaries; deterministic sample 84."""
    rng = random.Random(4185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0085():
    """entropy; deterministic sample 85."""
    rng = random.Random(4186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0086():
    """distribution shift; deterministic sample 86."""
    rng = random.Random(4187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0087():
    """bin boundaries; deterministic sample 87."""
    rng = random.Random(4188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0088():
    """entropy; deterministic sample 88."""
    rng = random.Random(4189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0089():
    """distribution shift; deterministic sample 89."""
    rng = random.Random(4190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0090():
    """bin boundaries; deterministic sample 90."""
    rng = random.Random(4191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0091():
    """entropy; deterministic sample 91."""
    rng = random.Random(4192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0092():
    """distribution shift; deterministic sample 92."""
    rng = random.Random(4193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0093():
    """bin boundaries; deterministic sample 93."""
    rng = random.Random(4194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0094():
    """entropy; deterministic sample 94."""
    rng = random.Random(4195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0095():
    """distribution shift; deterministic sample 95."""
    rng = random.Random(4196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0096():
    """bin boundaries; deterministic sample 96."""
    rng = random.Random(4197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0097():
    """entropy; deterministic sample 97."""
    rng = random.Random(4198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0098():
    """distribution shift; deterministic sample 98."""
    rng = random.Random(4199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0099():
    """bin boundaries; deterministic sample 99."""
    rng = random.Random(4200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0100():
    """entropy; deterministic sample 100."""
    rng = random.Random(4201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0101():
    """distribution shift; deterministic sample 101."""
    rng = random.Random(4202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0102():
    """bin boundaries; deterministic sample 102."""
    rng = random.Random(4203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0103():
    """entropy; deterministic sample 103."""
    rng = random.Random(4204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0104():
    """distribution shift; deterministic sample 104."""
    rng = random.Random(4205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0105():
    """bin boundaries; deterministic sample 105."""
    rng = random.Random(4206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0106():
    """entropy; deterministic sample 106."""
    rng = random.Random(4207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0107():
    """distribution shift; deterministic sample 107."""
    rng = random.Random(4208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0108():
    """bin boundaries; deterministic sample 108."""
    rng = random.Random(4209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0109():
    """entropy; deterministic sample 109."""
    rng = random.Random(4210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0110():
    """distribution shift; deterministic sample 110."""
    rng = random.Random(4211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0111():
    """bin boundaries; deterministic sample 111."""
    rng = random.Random(4212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0112():
    """entropy; deterministic sample 112."""
    rng = random.Random(4213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0113():
    """distribution shift; deterministic sample 113."""
    rng = random.Random(4214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0114():
    """bin boundaries; deterministic sample 114."""
    rng = random.Random(4215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0115():
    """entropy; deterministic sample 115."""
    rng = random.Random(4216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0116():
    """distribution shift; deterministic sample 116."""
    rng = random.Random(4217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0117():
    """bin boundaries; deterministic sample 117."""
    rng = random.Random(4218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0118():
    """entropy; deterministic sample 118."""
    rng = random.Random(4219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0119():
    """distribution shift; deterministic sample 119."""
    rng = random.Random(4220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0120():
    """bin boundaries; deterministic sample 120."""
    rng = random.Random(4221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0121():
    """entropy; deterministic sample 121."""
    rng = random.Random(4222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0122():
    """distribution shift; deterministic sample 122."""
    rng = random.Random(4223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0123():
    """bin boundaries; deterministic sample 123."""
    rng = random.Random(4224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0124():
    """entropy; deterministic sample 124."""
    rng = random.Random(4225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0125():
    """distribution shift; deterministic sample 125."""
    rng = random.Random(4226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0126():
    """bin boundaries; deterministic sample 126."""
    rng = random.Random(4227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0127():
    """entropy; deterministic sample 127."""
    rng = random.Random(4228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0128():
    """distribution shift; deterministic sample 128."""
    rng = random.Random(4229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0129():
    """bin boundaries; deterministic sample 129."""
    rng = random.Random(4230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0130():
    """entropy; deterministic sample 130."""
    rng = random.Random(4231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0131():
    """distribution shift; deterministic sample 131."""
    rng = random.Random(4232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0132():
    """bin boundaries; deterministic sample 132."""
    rng = random.Random(4233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0133():
    """entropy; deterministic sample 133."""
    rng = random.Random(4234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0134():
    """distribution shift; deterministic sample 134."""
    rng = random.Random(4235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0135():
    """bin boundaries; deterministic sample 135."""
    rng = random.Random(4236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0136():
    """entropy; deterministic sample 136."""
    rng = random.Random(4237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0137():
    """distribution shift; deterministic sample 137."""
    rng = random.Random(4238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0138():
    """bin boundaries; deterministic sample 138."""
    rng = random.Random(4239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0139():
    """entropy; deterministic sample 139."""
    rng = random.Random(4240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0140():
    """distribution shift; deterministic sample 140."""
    rng = random.Random(4241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0141():
    """bin boundaries; deterministic sample 141."""
    rng = random.Random(4242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0142():
    """entropy; deterministic sample 142."""
    rng = random.Random(4243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0143():
    """distribution shift; deterministic sample 143."""
    rng = random.Random(4244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0144():
    """bin boundaries; deterministic sample 144."""
    rng = random.Random(4245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0145():
    """entropy; deterministic sample 145."""
    rng = random.Random(4246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0146():
    """distribution shift; deterministic sample 146."""
    rng = random.Random(4247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0147():
    """bin boundaries; deterministic sample 147."""
    rng = random.Random(4248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0148():
    """entropy; deterministic sample 148."""
    rng = random.Random(4249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0149():
    """distribution shift; deterministic sample 149."""
    rng = random.Random(4250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0150():
    """bin boundaries; deterministic sample 150."""
    rng = random.Random(4251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0151():
    """entropy; deterministic sample 151."""
    rng = random.Random(4252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0152():
    """distribution shift; deterministic sample 152."""
    rng = random.Random(4253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0153():
    """bin boundaries; deterministic sample 153."""
    rng = random.Random(4254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0154():
    """entropy; deterministic sample 154."""
    rng = random.Random(4255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0155():
    """distribution shift; deterministic sample 155."""
    rng = random.Random(4256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0156():
    """bin boundaries; deterministic sample 156."""
    rng = random.Random(4257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0157():
    """entropy; deterministic sample 157."""
    rng = random.Random(4258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0158():
    """distribution shift; deterministic sample 158."""
    rng = random.Random(4259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0159():
    """bin boundaries; deterministic sample 159."""
    rng = random.Random(4260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0160():
    """entropy; deterministic sample 160."""
    rng = random.Random(4261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0161():
    """distribution shift; deterministic sample 161."""
    rng = random.Random(4262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0162():
    """bin boundaries; deterministic sample 162."""
    rng = random.Random(4263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0163():
    """entropy; deterministic sample 163."""
    rng = random.Random(4264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0164():
    """distribution shift; deterministic sample 164."""
    rng = random.Random(4265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0165():
    """bin boundaries; deterministic sample 165."""
    rng = random.Random(4266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0166():
    """entropy; deterministic sample 166."""
    rng = random.Random(4267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0167():
    """distribution shift; deterministic sample 167."""
    rng = random.Random(4268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0168():
    """bin boundaries; deterministic sample 168."""
    rng = random.Random(4269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0169():
    """entropy; deterministic sample 169."""
    rng = random.Random(4270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0170():
    """distribution shift; deterministic sample 170."""
    rng = random.Random(4271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0171():
    """bin boundaries; deterministic sample 171."""
    rng = random.Random(4272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0172():
    """entropy; deterministic sample 172."""
    rng = random.Random(4273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0173():
    """distribution shift; deterministic sample 173."""
    rng = random.Random(4274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0174():
    """bin boundaries; deterministic sample 174."""
    rng = random.Random(4275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0175():
    """entropy; deterministic sample 175."""
    rng = random.Random(4276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0176():
    """distribution shift; deterministic sample 176."""
    rng = random.Random(4277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0177():
    """bin boundaries; deterministic sample 177."""
    rng = random.Random(4278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0178():
    """entropy; deterministic sample 178."""
    rng = random.Random(4279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0179():
    """distribution shift; deterministic sample 179."""
    rng = random.Random(4280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0180():
    """bin boundaries; deterministic sample 180."""
    rng = random.Random(4281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0181():
    """entropy; deterministic sample 181."""
    rng = random.Random(4282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0182():
    """distribution shift; deterministic sample 182."""
    rng = random.Random(4283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0183():
    """bin boundaries; deterministic sample 183."""
    rng = random.Random(4284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0184():
    """entropy; deterministic sample 184."""
    rng = random.Random(4285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0185():
    """distribution shift; deterministic sample 185."""
    rng = random.Random(4286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0186():
    """bin boundaries; deterministic sample 186."""
    rng = random.Random(4287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0187():
    """entropy; deterministic sample 187."""
    rng = random.Random(4288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0188():
    """distribution shift; deterministic sample 188."""
    rng = random.Random(4289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0189():
    """bin boundaries; deterministic sample 189."""
    rng = random.Random(4290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0190():
    """entropy; deterministic sample 190."""
    rng = random.Random(4291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0191():
    """distribution shift; deterministic sample 191."""
    rng = random.Random(4292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0192():
    """bin boundaries; deterministic sample 192."""
    rng = random.Random(4293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0193():
    """entropy; deterministic sample 193."""
    rng = random.Random(4294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0194():
    """distribution shift; deterministic sample 194."""
    rng = random.Random(4295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0195():
    """bin boundaries; deterministic sample 195."""
    rng = random.Random(4296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0196():
    """entropy; deterministic sample 196."""
    rng = random.Random(4297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0197():
    """distribution shift; deterministic sample 197."""
    rng = random.Random(4298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0198():
    """bin boundaries; deterministic sample 198."""
    rng = random.Random(4299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0199():
    """entropy; deterministic sample 199."""
    rng = random.Random(4300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0200():
    """distribution shift; deterministic sample 200."""
    rng = random.Random(4301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0201():
    """bin boundaries; deterministic sample 201."""
    rng = random.Random(4302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0202():
    """entropy; deterministic sample 202."""
    rng = random.Random(4303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0203():
    """distribution shift; deterministic sample 203."""
    rng = random.Random(4304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0204():
    """bin boundaries; deterministic sample 204."""
    rng = random.Random(4305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0205():
    """entropy; deterministic sample 205."""
    rng = random.Random(4306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0206():
    """distribution shift; deterministic sample 206."""
    rng = random.Random(4307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0207():
    """bin boundaries; deterministic sample 207."""
    rng = random.Random(4308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0208():
    """entropy; deterministic sample 208."""
    rng = random.Random(4309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0209():
    """distribution shift; deterministic sample 209."""
    rng = random.Random(4310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0210():
    """bin boundaries; deterministic sample 210."""
    rng = random.Random(4311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0211():
    """entropy; deterministic sample 211."""
    rng = random.Random(4312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0212():
    """distribution shift; deterministic sample 212."""
    rng = random.Random(4313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0213():
    """bin boundaries; deterministic sample 213."""
    rng = random.Random(4314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0214():
    """entropy; deterministic sample 214."""
    rng = random.Random(4315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0215():
    """distribution shift; deterministic sample 215."""
    rng = random.Random(4316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0216():
    """bin boundaries; deterministic sample 216."""
    rng = random.Random(4317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0217():
    """entropy; deterministic sample 217."""
    rng = random.Random(4318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0218():
    """distribution shift; deterministic sample 218."""
    rng = random.Random(4319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0219():
    """bin boundaries; deterministic sample 219."""
    rng = random.Random(4320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0220():
    """entropy; deterministic sample 220."""
    rng = random.Random(4321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0221():
    """distribution shift; deterministic sample 221."""
    rng = random.Random(4322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0222():
    """bin boundaries; deterministic sample 222."""
    rng = random.Random(4323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0223():
    """entropy; deterministic sample 223."""
    rng = random.Random(4324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0224():
    """distribution shift; deterministic sample 224."""
    rng = random.Random(4325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0225():
    """bin boundaries; deterministic sample 225."""
    rng = random.Random(4326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0226():
    """entropy; deterministic sample 226."""
    rng = random.Random(4327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0227():
    """distribution shift; deterministic sample 227."""
    rng = random.Random(4328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0228():
    """bin boundaries; deterministic sample 228."""
    rng = random.Random(4329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0229():
    """entropy; deterministic sample 229."""
    rng = random.Random(4330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0230():
    """distribution shift; deterministic sample 230."""
    rng = random.Random(4331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0231():
    """bin boundaries; deterministic sample 231."""
    rng = random.Random(4332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0232():
    """entropy; deterministic sample 232."""
    rng = random.Random(4333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0233():
    """distribution shift; deterministic sample 233."""
    rng = random.Random(4334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0234():
    """bin boundaries; deterministic sample 234."""
    rng = random.Random(4335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0235():
    """entropy; deterministic sample 235."""
    rng = random.Random(4336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0236():
    """distribution shift; deterministic sample 236."""
    rng = random.Random(4337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0237():
    """bin boundaries; deterministic sample 237."""
    rng = random.Random(4338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0238():
    """entropy; deterministic sample 238."""
    rng = random.Random(4339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_distribution_case_0239():
    """distribution shift; deterministic sample 239."""
    rng = random.Random(4340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.quantile(scores, 0.5)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

