"""Regression cases for regression: deterministic scenarios and invariants."""
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

