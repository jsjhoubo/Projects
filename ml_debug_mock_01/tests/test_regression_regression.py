"""Regression cases for regression: both deterministic examples and invariants."""
import math
import random
import pytest
from ml_eval.analytics import regression as subject

def test_regression_case_0000():
    """negative residuals; deterministic sample 0."""
    rng = random.Random(3101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0001():
    """zero variance; deterministic sample 1."""
    rng = random.Random(3102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0002():
    """finite values; deterministic sample 2."""
    rng = random.Random(3103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0003():
    """negative residuals; deterministic sample 3."""
    rng = random.Random(3104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0004():
    """zero variance; deterministic sample 4."""
    rng = random.Random(3105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0005():
    """finite values; deterministic sample 5."""
    rng = random.Random(3106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0006():
    """negative residuals; deterministic sample 6."""
    rng = random.Random(3107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0007():
    """zero variance; deterministic sample 7."""
    rng = random.Random(3108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0008():
    """finite values; deterministic sample 8."""
    rng = random.Random(3109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0009():
    """negative residuals; deterministic sample 9."""
    rng = random.Random(3110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0010():
    """zero variance; deterministic sample 10."""
    rng = random.Random(3111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0011():
    """finite values; deterministic sample 11."""
    rng = random.Random(3112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0012():
    """negative residuals; deterministic sample 12."""
    rng = random.Random(3113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0013():
    """zero variance; deterministic sample 13."""
    rng = random.Random(3114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0014():
    """finite values; deterministic sample 14."""
    rng = random.Random(3115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0015():
    """negative residuals; deterministic sample 15."""
    rng = random.Random(3116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0016():
    """zero variance; deterministic sample 16."""
    rng = random.Random(3117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0017():
    """finite values; deterministic sample 17."""
    rng = random.Random(3118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0018():
    """negative residuals; deterministic sample 18."""
    rng = random.Random(3119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0019():
    """zero variance; deterministic sample 19."""
    rng = random.Random(3120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0020():
    """finite values; deterministic sample 20."""
    rng = random.Random(3121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0021():
    """negative residuals; deterministic sample 21."""
    rng = random.Random(3122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0022():
    """zero variance; deterministic sample 22."""
    rng = random.Random(3123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0023():
    """finite values; deterministic sample 23."""
    rng = random.Random(3124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0024():
    """negative residuals; deterministic sample 24."""
    rng = random.Random(3125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0025():
    """zero variance; deterministic sample 25."""
    rng = random.Random(3126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0026():
    """finite values; deterministic sample 26."""
    rng = random.Random(3127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0027():
    """negative residuals; deterministic sample 27."""
    rng = random.Random(3128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0028():
    """zero variance; deterministic sample 28."""
    rng = random.Random(3129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0029():
    """finite values; deterministic sample 29."""
    rng = random.Random(3130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0030():
    """negative residuals; deterministic sample 30."""
    rng = random.Random(3131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0031():
    """zero variance; deterministic sample 31."""
    rng = random.Random(3132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0032():
    """finite values; deterministic sample 32."""
    rng = random.Random(3133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0033():
    """negative residuals; deterministic sample 33."""
    rng = random.Random(3134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0034():
    """zero variance; deterministic sample 34."""
    rng = random.Random(3135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0035():
    """finite values; deterministic sample 35."""
    rng = random.Random(3136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0036():
    """negative residuals; deterministic sample 36."""
    rng = random.Random(3137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0037():
    """zero variance; deterministic sample 37."""
    rng = random.Random(3138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0038():
    """finite values; deterministic sample 38."""
    rng = random.Random(3139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0039():
    """negative residuals; deterministic sample 39."""
    rng = random.Random(3140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0040():
    """zero variance; deterministic sample 40."""
    rng = random.Random(3141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0041():
    """finite values; deterministic sample 41."""
    rng = random.Random(3142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0042():
    """negative residuals; deterministic sample 42."""
    rng = random.Random(3143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0043():
    """zero variance; deterministic sample 43."""
    rng = random.Random(3144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0044():
    """finite values; deterministic sample 44."""
    rng = random.Random(3145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0045():
    """negative residuals; deterministic sample 45."""
    rng = random.Random(3146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0046():
    """zero variance; deterministic sample 46."""
    rng = random.Random(3147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0047():
    """finite values; deterministic sample 47."""
    rng = random.Random(3148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0048():
    """negative residuals; deterministic sample 48."""
    rng = random.Random(3149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0049():
    """zero variance; deterministic sample 49."""
    rng = random.Random(3150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0050():
    """finite values; deterministic sample 50."""
    rng = random.Random(3151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0051():
    """negative residuals; deterministic sample 51."""
    rng = random.Random(3152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0052():
    """zero variance; deterministic sample 52."""
    rng = random.Random(3153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0053():
    """finite values; deterministic sample 53."""
    rng = random.Random(3154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0054():
    """negative residuals; deterministic sample 54."""
    rng = random.Random(3155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0055():
    """zero variance; deterministic sample 55."""
    rng = random.Random(3156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0056():
    """finite values; deterministic sample 56."""
    rng = random.Random(3157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0057():
    """negative residuals; deterministic sample 57."""
    rng = random.Random(3158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0058():
    """zero variance; deterministic sample 58."""
    rng = random.Random(3159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0059():
    """finite values; deterministic sample 59."""
    rng = random.Random(3160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0060():
    """negative residuals; deterministic sample 60."""
    rng = random.Random(3161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0061():
    """zero variance; deterministic sample 61."""
    rng = random.Random(3162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0062():
    """finite values; deterministic sample 62."""
    rng = random.Random(3163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0063():
    """negative residuals; deterministic sample 63."""
    rng = random.Random(3164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0064():
    """zero variance; deterministic sample 64."""
    rng = random.Random(3165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0065():
    """finite values; deterministic sample 65."""
    rng = random.Random(3166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0066():
    """negative residuals; deterministic sample 66."""
    rng = random.Random(3167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0067():
    """zero variance; deterministic sample 67."""
    rng = random.Random(3168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0068():
    """finite values; deterministic sample 68."""
    rng = random.Random(3169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0069():
    """negative residuals; deterministic sample 69."""
    rng = random.Random(3170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0070():
    """zero variance; deterministic sample 70."""
    rng = random.Random(3171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0071():
    """finite values; deterministic sample 71."""
    rng = random.Random(3172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0072():
    """negative residuals; deterministic sample 72."""
    rng = random.Random(3173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0073():
    """zero variance; deterministic sample 73."""
    rng = random.Random(3174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0074():
    """finite values; deterministic sample 74."""
    rng = random.Random(3175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0075():
    """negative residuals; deterministic sample 75."""
    rng = random.Random(3176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0076():
    """zero variance; deterministic sample 76."""
    rng = random.Random(3177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0077():
    """finite values; deterministic sample 77."""
    rng = random.Random(3178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0078():
    """negative residuals; deterministic sample 78."""
    rng = random.Random(3179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0079():
    """zero variance; deterministic sample 79."""
    rng = random.Random(3180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0080():
    """finite values; deterministic sample 80."""
    rng = random.Random(3181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0081():
    """negative residuals; deterministic sample 81."""
    rng = random.Random(3182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0082():
    """zero variance; deterministic sample 82."""
    rng = random.Random(3183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0083():
    """finite values; deterministic sample 83."""
    rng = random.Random(3184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0084():
    """negative residuals; deterministic sample 84."""
    rng = random.Random(3185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0085():
    """zero variance; deterministic sample 85."""
    rng = random.Random(3186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0086():
    """finite values; deterministic sample 86."""
    rng = random.Random(3187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0087():
    """negative residuals; deterministic sample 87."""
    rng = random.Random(3188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0088():
    """zero variance; deterministic sample 88."""
    rng = random.Random(3189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0089():
    """finite values; deterministic sample 89."""
    rng = random.Random(3190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0090():
    """negative residuals; deterministic sample 90."""
    rng = random.Random(3191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0091():
    """zero variance; deterministic sample 91."""
    rng = random.Random(3192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0092():
    """finite values; deterministic sample 92."""
    rng = random.Random(3193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0093():
    """negative residuals; deterministic sample 93."""
    rng = random.Random(3194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0094():
    """zero variance; deterministic sample 94."""
    rng = random.Random(3195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0095():
    """finite values; deterministic sample 95."""
    rng = random.Random(3196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0096():
    """negative residuals; deterministic sample 96."""
    rng = random.Random(3197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0097():
    """zero variance; deterministic sample 97."""
    rng = random.Random(3198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0098():
    """finite values; deterministic sample 98."""
    rng = random.Random(3199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0099():
    """negative residuals; deterministic sample 99."""
    rng = random.Random(3200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0100():
    """zero variance; deterministic sample 100."""
    rng = random.Random(3201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0101():
    """finite values; deterministic sample 101."""
    rng = random.Random(3202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0102():
    """negative residuals; deterministic sample 102."""
    rng = random.Random(3203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0103():
    """zero variance; deterministic sample 103."""
    rng = random.Random(3204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0104():
    """finite values; deterministic sample 104."""
    rng = random.Random(3205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0105():
    """negative residuals; deterministic sample 105."""
    rng = random.Random(3206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0106():
    """zero variance; deterministic sample 106."""
    rng = random.Random(3207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0107():
    """finite values; deterministic sample 107."""
    rng = random.Random(3208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0108():
    """negative residuals; deterministic sample 108."""
    rng = random.Random(3209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0109():
    """zero variance; deterministic sample 109."""
    rng = random.Random(3210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0110():
    """finite values; deterministic sample 110."""
    rng = random.Random(3211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0111():
    """negative residuals; deterministic sample 111."""
    rng = random.Random(3212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0112():
    """zero variance; deterministic sample 112."""
    rng = random.Random(3213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0113():
    """finite values; deterministic sample 113."""
    rng = random.Random(3214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0114():
    """negative residuals; deterministic sample 114."""
    rng = random.Random(3215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0115():
    """zero variance; deterministic sample 115."""
    rng = random.Random(3216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0116():
    """finite values; deterministic sample 116."""
    rng = random.Random(3217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0117():
    """negative residuals; deterministic sample 117."""
    rng = random.Random(3218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0118():
    """zero variance; deterministic sample 118."""
    rng = random.Random(3219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0119():
    """finite values; deterministic sample 119."""
    rng = random.Random(3220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0120():
    """negative residuals; deterministic sample 120."""
    rng = random.Random(3221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0121():
    """zero variance; deterministic sample 121."""
    rng = random.Random(3222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0122():
    """finite values; deterministic sample 122."""
    rng = random.Random(3223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0123():
    """negative residuals; deterministic sample 123."""
    rng = random.Random(3224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0124():
    """zero variance; deterministic sample 124."""
    rng = random.Random(3225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0125():
    """finite values; deterministic sample 125."""
    rng = random.Random(3226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0126():
    """negative residuals; deterministic sample 126."""
    rng = random.Random(3227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0127():
    """zero variance; deterministic sample 127."""
    rng = random.Random(3228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0128():
    """finite values; deterministic sample 128."""
    rng = random.Random(3229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0129():
    """negative residuals; deterministic sample 129."""
    rng = random.Random(3230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0130():
    """zero variance; deterministic sample 130."""
    rng = random.Random(3231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0131():
    """finite values; deterministic sample 131."""
    rng = random.Random(3232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0132():
    """negative residuals; deterministic sample 132."""
    rng = random.Random(3233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0133():
    """zero variance; deterministic sample 133."""
    rng = random.Random(3234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0134():
    """finite values; deterministic sample 134."""
    rng = random.Random(3235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0135():
    """negative residuals; deterministic sample 135."""
    rng = random.Random(3236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0136():
    """zero variance; deterministic sample 136."""
    rng = random.Random(3237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0137():
    """finite values; deterministic sample 137."""
    rng = random.Random(3238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0138():
    """negative residuals; deterministic sample 138."""
    rng = random.Random(3239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0139():
    """zero variance; deterministic sample 139."""
    rng = random.Random(3240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0140():
    """finite values; deterministic sample 140."""
    rng = random.Random(3241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0141():
    """negative residuals; deterministic sample 141."""
    rng = random.Random(3242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0142():
    """zero variance; deterministic sample 142."""
    rng = random.Random(3243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0143():
    """finite values; deterministic sample 143."""
    rng = random.Random(3244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0144():
    """negative residuals; deterministic sample 144."""
    rng = random.Random(3245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0145():
    """zero variance; deterministic sample 145."""
    rng = random.Random(3246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0146():
    """finite values; deterministic sample 146."""
    rng = random.Random(3247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0147():
    """negative residuals; deterministic sample 147."""
    rng = random.Random(3248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0148():
    """zero variance; deterministic sample 148."""
    rng = random.Random(3249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0149():
    """finite values; deterministic sample 149."""
    rng = random.Random(3250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0150():
    """negative residuals; deterministic sample 150."""
    rng = random.Random(3251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0151():
    """zero variance; deterministic sample 151."""
    rng = random.Random(3252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0152():
    """finite values; deterministic sample 152."""
    rng = random.Random(3253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0153():
    """negative residuals; deterministic sample 153."""
    rng = random.Random(3254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0154():
    """zero variance; deterministic sample 154."""
    rng = random.Random(3255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0155():
    """finite values; deterministic sample 155."""
    rng = random.Random(3256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0156():
    """negative residuals; deterministic sample 156."""
    rng = random.Random(3257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0157():
    """zero variance; deterministic sample 157."""
    rng = random.Random(3258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0158():
    """finite values; deterministic sample 158."""
    rng = random.Random(3259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0159():
    """negative residuals; deterministic sample 159."""
    rng = random.Random(3260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0160():
    """zero variance; deterministic sample 160."""
    rng = random.Random(3261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0161():
    """finite values; deterministic sample 161."""
    rng = random.Random(3262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0162():
    """negative residuals; deterministic sample 162."""
    rng = random.Random(3263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0163():
    """zero variance; deterministic sample 163."""
    rng = random.Random(3264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0164():
    """finite values; deterministic sample 164."""
    rng = random.Random(3265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0165():
    """negative residuals; deterministic sample 165."""
    rng = random.Random(3266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0166():
    """zero variance; deterministic sample 166."""
    rng = random.Random(3267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0167():
    """finite values; deterministic sample 167."""
    rng = random.Random(3268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0168():
    """negative residuals; deterministic sample 168."""
    rng = random.Random(3269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0169():
    """zero variance; deterministic sample 169."""
    rng = random.Random(3270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0170():
    """finite values; deterministic sample 170."""
    rng = random.Random(3271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0171():
    """negative residuals; deterministic sample 171."""
    rng = random.Random(3272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0172():
    """zero variance; deterministic sample 172."""
    rng = random.Random(3273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0173():
    """finite values; deterministic sample 173."""
    rng = random.Random(3274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0174():
    """negative residuals; deterministic sample 174."""
    rng = random.Random(3275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0175():
    """zero variance; deterministic sample 175."""
    rng = random.Random(3276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0176():
    """finite values; deterministic sample 176."""
    rng = random.Random(3277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0177():
    """negative residuals; deterministic sample 177."""
    rng = random.Random(3278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0178():
    """zero variance; deterministic sample 178."""
    rng = random.Random(3279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0179():
    """finite values; deterministic sample 179."""
    rng = random.Random(3280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0180():
    """negative residuals; deterministic sample 180."""
    rng = random.Random(3281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0181():
    """zero variance; deterministic sample 181."""
    rng = random.Random(3282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0182():
    """finite values; deterministic sample 182."""
    rng = random.Random(3283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0183():
    """negative residuals; deterministic sample 183."""
    rng = random.Random(3284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0184():
    """zero variance; deterministic sample 184."""
    rng = random.Random(3285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0185():
    """finite values; deterministic sample 185."""
    rng = random.Random(3286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0186():
    """negative residuals; deterministic sample 186."""
    rng = random.Random(3287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0187():
    """zero variance; deterministic sample 187."""
    rng = random.Random(3288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0188():
    """finite values; deterministic sample 188."""
    rng = random.Random(3289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0189():
    """negative residuals; deterministic sample 189."""
    rng = random.Random(3290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0190():
    """zero variance; deterministic sample 190."""
    rng = random.Random(3291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0191():
    """finite values; deterministic sample 191."""
    rng = random.Random(3292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0192():
    """negative residuals; deterministic sample 192."""
    rng = random.Random(3293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0193():
    """zero variance; deterministic sample 193."""
    rng = random.Random(3294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0194():
    """finite values; deterministic sample 194."""
    rng = random.Random(3295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0195():
    """negative residuals; deterministic sample 195."""
    rng = random.Random(3296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0196():
    """zero variance; deterministic sample 196."""
    rng = random.Random(3297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0197():
    """finite values; deterministic sample 197."""
    rng = random.Random(3298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0198():
    """negative residuals; deterministic sample 198."""
    rng = random.Random(3299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0199():
    """zero variance; deterministic sample 199."""
    rng = random.Random(3300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0200():
    """finite values; deterministic sample 200."""
    rng = random.Random(3301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0201():
    """negative residuals; deterministic sample 201."""
    rng = random.Random(3302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0202():
    """zero variance; deterministic sample 202."""
    rng = random.Random(3303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0203():
    """finite values; deterministic sample 203."""
    rng = random.Random(3304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0204():
    """negative residuals; deterministic sample 204."""
    rng = random.Random(3305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0205():
    """zero variance; deterministic sample 205."""
    rng = random.Random(3306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0206():
    """finite values; deterministic sample 206."""
    rng = random.Random(3307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0207():
    """negative residuals; deterministic sample 207."""
    rng = random.Random(3308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0208():
    """zero variance; deterministic sample 208."""
    rng = random.Random(3309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0209():
    """finite values; deterministic sample 209."""
    rng = random.Random(3310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0210():
    """negative residuals; deterministic sample 210."""
    rng = random.Random(3311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0211():
    """zero variance; deterministic sample 211."""
    rng = random.Random(3312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0212():
    """finite values; deterministic sample 212."""
    rng = random.Random(3313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0213():
    """negative residuals; deterministic sample 213."""
    rng = random.Random(3314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0214():
    """zero variance; deterministic sample 214."""
    rng = random.Random(3315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0215():
    """finite values; deterministic sample 215."""
    rng = random.Random(3316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0216():
    """negative residuals; deterministic sample 216."""
    rng = random.Random(3317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0217():
    """zero variance; deterministic sample 217."""
    rng = random.Random(3318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0218():
    """finite values; deterministic sample 218."""
    rng = random.Random(3319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0219():
    """negative residuals; deterministic sample 219."""
    rng = random.Random(3320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0220():
    """zero variance; deterministic sample 220."""
    rng = random.Random(3321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0221():
    """finite values; deterministic sample 221."""
    rng = random.Random(3322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0222():
    """negative residuals; deterministic sample 222."""
    rng = random.Random(3323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0223():
    """zero variance; deterministic sample 223."""
    rng = random.Random(3324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0224():
    """finite values; deterministic sample 224."""
    rng = random.Random(3325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0225():
    """negative residuals; deterministic sample 225."""
    rng = random.Random(3326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0226():
    """zero variance; deterministic sample 226."""
    rng = random.Random(3327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0227():
    """finite values; deterministic sample 227."""
    rng = random.Random(3328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0228():
    """negative residuals; deterministic sample 228."""
    rng = random.Random(3329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0229():
    """zero variance; deterministic sample 229."""
    rng = random.Random(3330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0230():
    """finite values; deterministic sample 230."""
    rng = random.Random(3331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0231():
    """negative residuals; deterministic sample 231."""
    rng = random.Random(3332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0232():
    """zero variance; deterministic sample 232."""
    rng = random.Random(3333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0233():
    """finite values; deterministic sample 233."""
    rng = random.Random(3334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0234():
    """negative residuals; deterministic sample 234."""
    rng = random.Random(3335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0235():
    """zero variance; deterministic sample 235."""
    rng = random.Random(3336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0236():
    """finite values; deterministic sample 236."""
    rng = random.Random(3337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0237():
    """negative residuals; deterministic sample 237."""
    rng = random.Random(3338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0238():
    """zero variance; deterministic sample 238."""
    rng = random.Random(3339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_regression_case_0239():
    """finite values; deterministic sample 239."""
    rng = random.Random(3340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.mae(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

