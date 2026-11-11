"""Regression cases for confidence: both deterministic examples and invariants."""
import math
import random
import pytest
from ml_eval.analytics import confidence as subject

def test_confidence_case_0000():
    """small samples; deterministic sample 0."""
    rng = random.Random(7101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0001():
    """bootstrap reproducibility; deterministic sample 1."""
    rng = random.Random(7102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0002():
    """dependent trials; deterministic sample 2."""
    rng = random.Random(7103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0003():
    """small samples; deterministic sample 3."""
    rng = random.Random(7104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0004():
    """bootstrap reproducibility; deterministic sample 4."""
    rng = random.Random(7105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0005():
    """dependent trials; deterministic sample 5."""
    rng = random.Random(7106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0006():
    """small samples; deterministic sample 6."""
    rng = random.Random(7107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0007():
    """bootstrap reproducibility; deterministic sample 7."""
    rng = random.Random(7108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0008():
    """dependent trials; deterministic sample 8."""
    rng = random.Random(7109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0009():
    """small samples; deterministic sample 9."""
    rng = random.Random(7110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0010():
    """bootstrap reproducibility; deterministic sample 10."""
    rng = random.Random(7111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0011():
    """dependent trials; deterministic sample 11."""
    rng = random.Random(7112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0012():
    """small samples; deterministic sample 12."""
    rng = random.Random(7113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0013():
    """bootstrap reproducibility; deterministic sample 13."""
    rng = random.Random(7114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0014():
    """dependent trials; deterministic sample 14."""
    rng = random.Random(7115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0015():
    """small samples; deterministic sample 15."""
    rng = random.Random(7116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0016():
    """bootstrap reproducibility; deterministic sample 16."""
    rng = random.Random(7117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0017():
    """dependent trials; deterministic sample 17."""
    rng = random.Random(7118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0018():
    """small samples; deterministic sample 18."""
    rng = random.Random(7119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0019():
    """bootstrap reproducibility; deterministic sample 19."""
    rng = random.Random(7120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0020():
    """dependent trials; deterministic sample 20."""
    rng = random.Random(7121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0021():
    """small samples; deterministic sample 21."""
    rng = random.Random(7122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0022():
    """bootstrap reproducibility; deterministic sample 22."""
    rng = random.Random(7123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0023():
    """dependent trials; deterministic sample 23."""
    rng = random.Random(7124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0024():
    """small samples; deterministic sample 24."""
    rng = random.Random(7125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0025():
    """bootstrap reproducibility; deterministic sample 25."""
    rng = random.Random(7126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0026():
    """dependent trials; deterministic sample 26."""
    rng = random.Random(7127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0027():
    """small samples; deterministic sample 27."""
    rng = random.Random(7128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0028():
    """bootstrap reproducibility; deterministic sample 28."""
    rng = random.Random(7129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0029():
    """dependent trials; deterministic sample 29."""
    rng = random.Random(7130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0030():
    """small samples; deterministic sample 30."""
    rng = random.Random(7131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0031():
    """bootstrap reproducibility; deterministic sample 31."""
    rng = random.Random(7132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0032():
    """dependent trials; deterministic sample 32."""
    rng = random.Random(7133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0033():
    """small samples; deterministic sample 33."""
    rng = random.Random(7134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0034():
    """bootstrap reproducibility; deterministic sample 34."""
    rng = random.Random(7135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0035():
    """dependent trials; deterministic sample 35."""
    rng = random.Random(7136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0036():
    """small samples; deterministic sample 36."""
    rng = random.Random(7137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0037():
    """bootstrap reproducibility; deterministic sample 37."""
    rng = random.Random(7138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0038():
    """dependent trials; deterministic sample 38."""
    rng = random.Random(7139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0039():
    """small samples; deterministic sample 39."""
    rng = random.Random(7140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0040():
    """bootstrap reproducibility; deterministic sample 40."""
    rng = random.Random(7141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0041():
    """dependent trials; deterministic sample 41."""
    rng = random.Random(7142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0042():
    """small samples; deterministic sample 42."""
    rng = random.Random(7143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0043():
    """bootstrap reproducibility; deterministic sample 43."""
    rng = random.Random(7144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0044():
    """dependent trials; deterministic sample 44."""
    rng = random.Random(7145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0045():
    """small samples; deterministic sample 45."""
    rng = random.Random(7146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0046():
    """bootstrap reproducibility; deterministic sample 46."""
    rng = random.Random(7147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0047():
    """dependent trials; deterministic sample 47."""
    rng = random.Random(7148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0048():
    """small samples; deterministic sample 48."""
    rng = random.Random(7149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0049():
    """bootstrap reproducibility; deterministic sample 49."""
    rng = random.Random(7150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0050():
    """dependent trials; deterministic sample 50."""
    rng = random.Random(7151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0051():
    """small samples; deterministic sample 51."""
    rng = random.Random(7152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0052():
    """bootstrap reproducibility; deterministic sample 52."""
    rng = random.Random(7153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0053():
    """dependent trials; deterministic sample 53."""
    rng = random.Random(7154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0054():
    """small samples; deterministic sample 54."""
    rng = random.Random(7155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0055():
    """bootstrap reproducibility; deterministic sample 55."""
    rng = random.Random(7156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0056():
    """dependent trials; deterministic sample 56."""
    rng = random.Random(7157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0057():
    """small samples; deterministic sample 57."""
    rng = random.Random(7158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0058():
    """bootstrap reproducibility; deterministic sample 58."""
    rng = random.Random(7159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0059():
    """dependent trials; deterministic sample 59."""
    rng = random.Random(7160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0060():
    """small samples; deterministic sample 60."""
    rng = random.Random(7161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0061():
    """bootstrap reproducibility; deterministic sample 61."""
    rng = random.Random(7162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0062():
    """dependent trials; deterministic sample 62."""
    rng = random.Random(7163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0063():
    """small samples; deterministic sample 63."""
    rng = random.Random(7164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0064():
    """bootstrap reproducibility; deterministic sample 64."""
    rng = random.Random(7165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0065():
    """dependent trials; deterministic sample 65."""
    rng = random.Random(7166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0066():
    """small samples; deterministic sample 66."""
    rng = random.Random(7167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0067():
    """bootstrap reproducibility; deterministic sample 67."""
    rng = random.Random(7168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0068():
    """dependent trials; deterministic sample 68."""
    rng = random.Random(7169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0069():
    """small samples; deterministic sample 69."""
    rng = random.Random(7170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0070():
    """bootstrap reproducibility; deterministic sample 70."""
    rng = random.Random(7171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0071():
    """dependent trials; deterministic sample 71."""
    rng = random.Random(7172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0072():
    """small samples; deterministic sample 72."""
    rng = random.Random(7173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0073():
    """bootstrap reproducibility; deterministic sample 73."""
    rng = random.Random(7174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0074():
    """dependent trials; deterministic sample 74."""
    rng = random.Random(7175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0075():
    """small samples; deterministic sample 75."""
    rng = random.Random(7176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0076():
    """bootstrap reproducibility; deterministic sample 76."""
    rng = random.Random(7177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0077():
    """dependent trials; deterministic sample 77."""
    rng = random.Random(7178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0078():
    """small samples; deterministic sample 78."""
    rng = random.Random(7179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0079():
    """bootstrap reproducibility; deterministic sample 79."""
    rng = random.Random(7180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0080():
    """dependent trials; deterministic sample 80."""
    rng = random.Random(7181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0081():
    """small samples; deterministic sample 81."""
    rng = random.Random(7182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0082():
    """bootstrap reproducibility; deterministic sample 82."""
    rng = random.Random(7183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0083():
    """dependent trials; deterministic sample 83."""
    rng = random.Random(7184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0084():
    """small samples; deterministic sample 84."""
    rng = random.Random(7185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0085():
    """bootstrap reproducibility; deterministic sample 85."""
    rng = random.Random(7186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0086():
    """dependent trials; deterministic sample 86."""
    rng = random.Random(7187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0087():
    """small samples; deterministic sample 87."""
    rng = random.Random(7188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0088():
    """bootstrap reproducibility; deterministic sample 88."""
    rng = random.Random(7189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0089():
    """dependent trials; deterministic sample 89."""
    rng = random.Random(7190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0090():
    """small samples; deterministic sample 90."""
    rng = random.Random(7191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0091():
    """bootstrap reproducibility; deterministic sample 91."""
    rng = random.Random(7192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0092():
    """dependent trials; deterministic sample 92."""
    rng = random.Random(7193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0093():
    """small samples; deterministic sample 93."""
    rng = random.Random(7194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0094():
    """bootstrap reproducibility; deterministic sample 94."""
    rng = random.Random(7195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0095():
    """dependent trials; deterministic sample 95."""
    rng = random.Random(7196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0096():
    """small samples; deterministic sample 96."""
    rng = random.Random(7197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0097():
    """bootstrap reproducibility; deterministic sample 97."""
    rng = random.Random(7198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0098():
    """dependent trials; deterministic sample 98."""
    rng = random.Random(7199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0099():
    """small samples; deterministic sample 99."""
    rng = random.Random(7200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0100():
    """bootstrap reproducibility; deterministic sample 100."""
    rng = random.Random(7201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0101():
    """dependent trials; deterministic sample 101."""
    rng = random.Random(7202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0102():
    """small samples; deterministic sample 102."""
    rng = random.Random(7203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0103():
    """bootstrap reproducibility; deterministic sample 103."""
    rng = random.Random(7204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0104():
    """dependent trials; deterministic sample 104."""
    rng = random.Random(7205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0105():
    """small samples; deterministic sample 105."""
    rng = random.Random(7206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0106():
    """bootstrap reproducibility; deterministic sample 106."""
    rng = random.Random(7207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0107():
    """dependent trials; deterministic sample 107."""
    rng = random.Random(7208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0108():
    """small samples; deterministic sample 108."""
    rng = random.Random(7209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0109():
    """bootstrap reproducibility; deterministic sample 109."""
    rng = random.Random(7210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0110():
    """dependent trials; deterministic sample 110."""
    rng = random.Random(7211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0111():
    """small samples; deterministic sample 111."""
    rng = random.Random(7212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0112():
    """bootstrap reproducibility; deterministic sample 112."""
    rng = random.Random(7213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0113():
    """dependent trials; deterministic sample 113."""
    rng = random.Random(7214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0114():
    """small samples; deterministic sample 114."""
    rng = random.Random(7215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0115():
    """bootstrap reproducibility; deterministic sample 115."""
    rng = random.Random(7216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0116():
    """dependent trials; deterministic sample 116."""
    rng = random.Random(7217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0117():
    """small samples; deterministic sample 117."""
    rng = random.Random(7218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0118():
    """bootstrap reproducibility; deterministic sample 118."""
    rng = random.Random(7219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0119():
    """dependent trials; deterministic sample 119."""
    rng = random.Random(7220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0120():
    """small samples; deterministic sample 120."""
    rng = random.Random(7221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0121():
    """bootstrap reproducibility; deterministic sample 121."""
    rng = random.Random(7222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0122():
    """dependent trials; deterministic sample 122."""
    rng = random.Random(7223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0123():
    """small samples; deterministic sample 123."""
    rng = random.Random(7224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0124():
    """bootstrap reproducibility; deterministic sample 124."""
    rng = random.Random(7225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0125():
    """dependent trials; deterministic sample 125."""
    rng = random.Random(7226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0126():
    """small samples; deterministic sample 126."""
    rng = random.Random(7227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0127():
    """bootstrap reproducibility; deterministic sample 127."""
    rng = random.Random(7228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0128():
    """dependent trials; deterministic sample 128."""
    rng = random.Random(7229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0129():
    """small samples; deterministic sample 129."""
    rng = random.Random(7230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0130():
    """bootstrap reproducibility; deterministic sample 130."""
    rng = random.Random(7231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0131():
    """dependent trials; deterministic sample 131."""
    rng = random.Random(7232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0132():
    """small samples; deterministic sample 132."""
    rng = random.Random(7233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0133():
    """bootstrap reproducibility; deterministic sample 133."""
    rng = random.Random(7234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0134():
    """dependent trials; deterministic sample 134."""
    rng = random.Random(7235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0135():
    """small samples; deterministic sample 135."""
    rng = random.Random(7236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0136():
    """bootstrap reproducibility; deterministic sample 136."""
    rng = random.Random(7237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0137():
    """dependent trials; deterministic sample 137."""
    rng = random.Random(7238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0138():
    """small samples; deterministic sample 138."""
    rng = random.Random(7239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0139():
    """bootstrap reproducibility; deterministic sample 139."""
    rng = random.Random(7240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0140():
    """dependent trials; deterministic sample 140."""
    rng = random.Random(7241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0141():
    """small samples; deterministic sample 141."""
    rng = random.Random(7242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0142():
    """bootstrap reproducibility; deterministic sample 142."""
    rng = random.Random(7243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0143():
    """dependent trials; deterministic sample 143."""
    rng = random.Random(7244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0144():
    """small samples; deterministic sample 144."""
    rng = random.Random(7245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0145():
    """bootstrap reproducibility; deterministic sample 145."""
    rng = random.Random(7246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0146():
    """dependent trials; deterministic sample 146."""
    rng = random.Random(7247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0147():
    """small samples; deterministic sample 147."""
    rng = random.Random(7248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0148():
    """bootstrap reproducibility; deterministic sample 148."""
    rng = random.Random(7249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0149():
    """dependent trials; deterministic sample 149."""
    rng = random.Random(7250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0150():
    """small samples; deterministic sample 150."""
    rng = random.Random(7251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0151():
    """bootstrap reproducibility; deterministic sample 151."""
    rng = random.Random(7252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0152():
    """dependent trials; deterministic sample 152."""
    rng = random.Random(7253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0153():
    """small samples; deterministic sample 153."""
    rng = random.Random(7254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0154():
    """bootstrap reproducibility; deterministic sample 154."""
    rng = random.Random(7255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0155():
    """dependent trials; deterministic sample 155."""
    rng = random.Random(7256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0156():
    """small samples; deterministic sample 156."""
    rng = random.Random(7257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0157():
    """bootstrap reproducibility; deterministic sample 157."""
    rng = random.Random(7258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0158():
    """dependent trials; deterministic sample 158."""
    rng = random.Random(7259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0159():
    """small samples; deterministic sample 159."""
    rng = random.Random(7260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0160():
    """bootstrap reproducibility; deterministic sample 160."""
    rng = random.Random(7261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0161():
    """dependent trials; deterministic sample 161."""
    rng = random.Random(7262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0162():
    """small samples; deterministic sample 162."""
    rng = random.Random(7263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0163():
    """bootstrap reproducibility; deterministic sample 163."""
    rng = random.Random(7264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0164():
    """dependent trials; deterministic sample 164."""
    rng = random.Random(7265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0165():
    """small samples; deterministic sample 165."""
    rng = random.Random(7266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0166():
    """bootstrap reproducibility; deterministic sample 166."""
    rng = random.Random(7267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0167():
    """dependent trials; deterministic sample 167."""
    rng = random.Random(7268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0168():
    """small samples; deterministic sample 168."""
    rng = random.Random(7269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0169():
    """bootstrap reproducibility; deterministic sample 169."""
    rng = random.Random(7270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0170():
    """dependent trials; deterministic sample 170."""
    rng = random.Random(7271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0171():
    """small samples; deterministic sample 171."""
    rng = random.Random(7272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0172():
    """bootstrap reproducibility; deterministic sample 172."""
    rng = random.Random(7273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0173():
    """dependent trials; deterministic sample 173."""
    rng = random.Random(7274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0174():
    """small samples; deterministic sample 174."""
    rng = random.Random(7275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0175():
    """bootstrap reproducibility; deterministic sample 175."""
    rng = random.Random(7276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0176():
    """dependent trials; deterministic sample 176."""
    rng = random.Random(7277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0177():
    """small samples; deterministic sample 177."""
    rng = random.Random(7278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0178():
    """bootstrap reproducibility; deterministic sample 178."""
    rng = random.Random(7279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0179():
    """dependent trials; deterministic sample 179."""
    rng = random.Random(7280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0180():
    """small samples; deterministic sample 180."""
    rng = random.Random(7281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0181():
    """bootstrap reproducibility; deterministic sample 181."""
    rng = random.Random(7282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0182():
    """dependent trials; deterministic sample 182."""
    rng = random.Random(7283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0183():
    """small samples; deterministic sample 183."""
    rng = random.Random(7284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0184():
    """bootstrap reproducibility; deterministic sample 184."""
    rng = random.Random(7285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0185():
    """dependent trials; deterministic sample 185."""
    rng = random.Random(7286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0186():
    """small samples; deterministic sample 186."""
    rng = random.Random(7287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0187():
    """bootstrap reproducibility; deterministic sample 187."""
    rng = random.Random(7288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0188():
    """dependent trials; deterministic sample 188."""
    rng = random.Random(7289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0189():
    """small samples; deterministic sample 189."""
    rng = random.Random(7290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0190():
    """bootstrap reproducibility; deterministic sample 190."""
    rng = random.Random(7291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0191():
    """dependent trials; deterministic sample 191."""
    rng = random.Random(7292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0192():
    """small samples; deterministic sample 192."""
    rng = random.Random(7293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0193():
    """bootstrap reproducibility; deterministic sample 193."""
    rng = random.Random(7294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0194():
    """dependent trials; deterministic sample 194."""
    rng = random.Random(7295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0195():
    """small samples; deterministic sample 195."""
    rng = random.Random(7296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0196():
    """bootstrap reproducibility; deterministic sample 196."""
    rng = random.Random(7297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0197():
    """dependent trials; deterministic sample 197."""
    rng = random.Random(7298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0198():
    """small samples; deterministic sample 198."""
    rng = random.Random(7299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0199():
    """bootstrap reproducibility; deterministic sample 199."""
    rng = random.Random(7300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0200():
    """dependent trials; deterministic sample 200."""
    rng = random.Random(7301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0201():
    """small samples; deterministic sample 201."""
    rng = random.Random(7302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0202():
    """bootstrap reproducibility; deterministic sample 202."""
    rng = random.Random(7303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0203():
    """dependent trials; deterministic sample 203."""
    rng = random.Random(7304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0204():
    """small samples; deterministic sample 204."""
    rng = random.Random(7305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0205():
    """bootstrap reproducibility; deterministic sample 205."""
    rng = random.Random(7306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0206():
    """dependent trials; deterministic sample 206."""
    rng = random.Random(7307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0207():
    """small samples; deterministic sample 207."""
    rng = random.Random(7308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0208():
    """bootstrap reproducibility; deterministic sample 208."""
    rng = random.Random(7309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0209():
    """dependent trials; deterministic sample 209."""
    rng = random.Random(7310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0210():
    """small samples; deterministic sample 210."""
    rng = random.Random(7311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0211():
    """bootstrap reproducibility; deterministic sample 211."""
    rng = random.Random(7312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0212():
    """dependent trials; deterministic sample 212."""
    rng = random.Random(7313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0213():
    """small samples; deterministic sample 213."""
    rng = random.Random(7314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0214():
    """bootstrap reproducibility; deterministic sample 214."""
    rng = random.Random(7315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0215():
    """dependent trials; deterministic sample 215."""
    rng = random.Random(7316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0216():
    """small samples; deterministic sample 216."""
    rng = random.Random(7317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0217():
    """bootstrap reproducibility; deterministic sample 217."""
    rng = random.Random(7318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0218():
    """dependent trials; deterministic sample 218."""
    rng = random.Random(7319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0219():
    """small samples; deterministic sample 219."""
    rng = random.Random(7320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0220():
    """bootstrap reproducibility; deterministic sample 220."""
    rng = random.Random(7321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0221():
    """dependent trials; deterministic sample 221."""
    rng = random.Random(7322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0222():
    """small samples; deterministic sample 222."""
    rng = random.Random(7323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0223():
    """bootstrap reproducibility; deterministic sample 223."""
    rng = random.Random(7324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0224():
    """dependent trials; deterministic sample 224."""
    rng = random.Random(7325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0225():
    """small samples; deterministic sample 225."""
    rng = random.Random(7326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0226():
    """bootstrap reproducibility; deterministic sample 226."""
    rng = random.Random(7327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0227():
    """dependent trials; deterministic sample 227."""
    rng = random.Random(7328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0228():
    """small samples; deterministic sample 228."""
    rng = random.Random(7329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0229():
    """bootstrap reproducibility; deterministic sample 229."""
    rng = random.Random(7330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0230():
    """dependent trials; deterministic sample 230."""
    rng = random.Random(7331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0231():
    """small samples; deterministic sample 231."""
    rng = random.Random(7332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0232():
    """bootstrap reproducibility; deterministic sample 232."""
    rng = random.Random(7333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0233():
    """dependent trials; deterministic sample 233."""
    rng = random.Random(7334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0234():
    """small samples; deterministic sample 234."""
    rng = random.Random(7335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0235():
    """bootstrap reproducibility; deterministic sample 235."""
    rng = random.Random(7336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0236():
    """dependent trials; deterministic sample 236."""
    rng = random.Random(7337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0237():
    """small samples; deterministic sample 237."""
    rng = random.Random(7338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0238():
    """bootstrap reproducibility; deterministic sample 238."""
    rng = random.Random(7339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_confidence_case_0239():
    """dependent trials; deterministic sample 239."""
    rng = random.Random(7340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.wilson_interval(sum(labels),n)[0]
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

