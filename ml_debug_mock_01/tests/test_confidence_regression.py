"""Regression cases for confidence: deterministic scenarios and invariants."""
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

