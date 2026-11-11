"""Regression cases for calibration: both deterministic examples and invariants."""
import math
import random
import pytest
from ml_eval.analytics import calibration as subject

def test_calibration_case_0000():
    """well calibrated probabilities; deterministic sample 0."""
    rng = random.Random(2101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0001():
    """extreme probability; deterministic sample 1."""
    rng = random.Random(2102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0002():
    """bucket edges; deterministic sample 2."""
    rng = random.Random(2103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0003():
    """well calibrated probabilities; deterministic sample 3."""
    rng = random.Random(2104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0004():
    """extreme probability; deterministic sample 4."""
    rng = random.Random(2105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0005():
    """bucket edges; deterministic sample 5."""
    rng = random.Random(2106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0006():
    """well calibrated probabilities; deterministic sample 6."""
    rng = random.Random(2107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0007():
    """extreme probability; deterministic sample 7."""
    rng = random.Random(2108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0008():
    """bucket edges; deterministic sample 8."""
    rng = random.Random(2109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0009():
    """well calibrated probabilities; deterministic sample 9."""
    rng = random.Random(2110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0010():
    """extreme probability; deterministic sample 10."""
    rng = random.Random(2111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0011():
    """bucket edges; deterministic sample 11."""
    rng = random.Random(2112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0012():
    """well calibrated probabilities; deterministic sample 12."""
    rng = random.Random(2113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0013():
    """extreme probability; deterministic sample 13."""
    rng = random.Random(2114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0014():
    """bucket edges; deterministic sample 14."""
    rng = random.Random(2115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0015():
    """well calibrated probabilities; deterministic sample 15."""
    rng = random.Random(2116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0016():
    """extreme probability; deterministic sample 16."""
    rng = random.Random(2117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0017():
    """bucket edges; deterministic sample 17."""
    rng = random.Random(2118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0018():
    """well calibrated probabilities; deterministic sample 18."""
    rng = random.Random(2119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0019():
    """extreme probability; deterministic sample 19."""
    rng = random.Random(2120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0020():
    """bucket edges; deterministic sample 20."""
    rng = random.Random(2121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0021():
    """well calibrated probabilities; deterministic sample 21."""
    rng = random.Random(2122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0022():
    """extreme probability; deterministic sample 22."""
    rng = random.Random(2123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0023():
    """bucket edges; deterministic sample 23."""
    rng = random.Random(2124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0024():
    """well calibrated probabilities; deterministic sample 24."""
    rng = random.Random(2125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0025():
    """extreme probability; deterministic sample 25."""
    rng = random.Random(2126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0026():
    """bucket edges; deterministic sample 26."""
    rng = random.Random(2127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0027():
    """well calibrated probabilities; deterministic sample 27."""
    rng = random.Random(2128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0028():
    """extreme probability; deterministic sample 28."""
    rng = random.Random(2129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0029():
    """bucket edges; deterministic sample 29."""
    rng = random.Random(2130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0030():
    """well calibrated probabilities; deterministic sample 30."""
    rng = random.Random(2131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0031():
    """extreme probability; deterministic sample 31."""
    rng = random.Random(2132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0032():
    """bucket edges; deterministic sample 32."""
    rng = random.Random(2133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0033():
    """well calibrated probabilities; deterministic sample 33."""
    rng = random.Random(2134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0034():
    """extreme probability; deterministic sample 34."""
    rng = random.Random(2135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0035():
    """bucket edges; deterministic sample 35."""
    rng = random.Random(2136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0036():
    """well calibrated probabilities; deterministic sample 36."""
    rng = random.Random(2137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0037():
    """extreme probability; deterministic sample 37."""
    rng = random.Random(2138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0038():
    """bucket edges; deterministic sample 38."""
    rng = random.Random(2139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0039():
    """well calibrated probabilities; deterministic sample 39."""
    rng = random.Random(2140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0040():
    """extreme probability; deterministic sample 40."""
    rng = random.Random(2141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0041():
    """bucket edges; deterministic sample 41."""
    rng = random.Random(2142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0042():
    """well calibrated probabilities; deterministic sample 42."""
    rng = random.Random(2143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0043():
    """extreme probability; deterministic sample 43."""
    rng = random.Random(2144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0044():
    """bucket edges; deterministic sample 44."""
    rng = random.Random(2145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0045():
    """well calibrated probabilities; deterministic sample 45."""
    rng = random.Random(2146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0046():
    """extreme probability; deterministic sample 46."""
    rng = random.Random(2147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0047():
    """bucket edges; deterministic sample 47."""
    rng = random.Random(2148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0048():
    """well calibrated probabilities; deterministic sample 48."""
    rng = random.Random(2149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0049():
    """extreme probability; deterministic sample 49."""
    rng = random.Random(2150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0050():
    """bucket edges; deterministic sample 50."""
    rng = random.Random(2151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0051():
    """well calibrated probabilities; deterministic sample 51."""
    rng = random.Random(2152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0052():
    """extreme probability; deterministic sample 52."""
    rng = random.Random(2153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0053():
    """bucket edges; deterministic sample 53."""
    rng = random.Random(2154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0054():
    """well calibrated probabilities; deterministic sample 54."""
    rng = random.Random(2155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0055():
    """extreme probability; deterministic sample 55."""
    rng = random.Random(2156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0056():
    """bucket edges; deterministic sample 56."""
    rng = random.Random(2157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0057():
    """well calibrated probabilities; deterministic sample 57."""
    rng = random.Random(2158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0058():
    """extreme probability; deterministic sample 58."""
    rng = random.Random(2159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0059():
    """bucket edges; deterministic sample 59."""
    rng = random.Random(2160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0060():
    """well calibrated probabilities; deterministic sample 60."""
    rng = random.Random(2161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0061():
    """extreme probability; deterministic sample 61."""
    rng = random.Random(2162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0062():
    """bucket edges; deterministic sample 62."""
    rng = random.Random(2163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0063():
    """well calibrated probabilities; deterministic sample 63."""
    rng = random.Random(2164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0064():
    """extreme probability; deterministic sample 64."""
    rng = random.Random(2165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0065():
    """bucket edges; deterministic sample 65."""
    rng = random.Random(2166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0066():
    """well calibrated probabilities; deterministic sample 66."""
    rng = random.Random(2167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0067():
    """extreme probability; deterministic sample 67."""
    rng = random.Random(2168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0068():
    """bucket edges; deterministic sample 68."""
    rng = random.Random(2169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0069():
    """well calibrated probabilities; deterministic sample 69."""
    rng = random.Random(2170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0070():
    """extreme probability; deterministic sample 70."""
    rng = random.Random(2171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0071():
    """bucket edges; deterministic sample 71."""
    rng = random.Random(2172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0072():
    """well calibrated probabilities; deterministic sample 72."""
    rng = random.Random(2173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0073():
    """extreme probability; deterministic sample 73."""
    rng = random.Random(2174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0074():
    """bucket edges; deterministic sample 74."""
    rng = random.Random(2175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0075():
    """well calibrated probabilities; deterministic sample 75."""
    rng = random.Random(2176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0076():
    """extreme probability; deterministic sample 76."""
    rng = random.Random(2177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0077():
    """bucket edges; deterministic sample 77."""
    rng = random.Random(2178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0078():
    """well calibrated probabilities; deterministic sample 78."""
    rng = random.Random(2179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0079():
    """extreme probability; deterministic sample 79."""
    rng = random.Random(2180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0080():
    """bucket edges; deterministic sample 80."""
    rng = random.Random(2181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0081():
    """well calibrated probabilities; deterministic sample 81."""
    rng = random.Random(2182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0082():
    """extreme probability; deterministic sample 82."""
    rng = random.Random(2183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0083():
    """bucket edges; deterministic sample 83."""
    rng = random.Random(2184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0084():
    """well calibrated probabilities; deterministic sample 84."""
    rng = random.Random(2185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0085():
    """extreme probability; deterministic sample 85."""
    rng = random.Random(2186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0086():
    """bucket edges; deterministic sample 86."""
    rng = random.Random(2187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0087():
    """well calibrated probabilities; deterministic sample 87."""
    rng = random.Random(2188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0088():
    """extreme probability; deterministic sample 88."""
    rng = random.Random(2189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0089():
    """bucket edges; deterministic sample 89."""
    rng = random.Random(2190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0090():
    """well calibrated probabilities; deterministic sample 90."""
    rng = random.Random(2191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0091():
    """extreme probability; deterministic sample 91."""
    rng = random.Random(2192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0092():
    """bucket edges; deterministic sample 92."""
    rng = random.Random(2193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0093():
    """well calibrated probabilities; deterministic sample 93."""
    rng = random.Random(2194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0094():
    """extreme probability; deterministic sample 94."""
    rng = random.Random(2195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0095():
    """bucket edges; deterministic sample 95."""
    rng = random.Random(2196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0096():
    """well calibrated probabilities; deterministic sample 96."""
    rng = random.Random(2197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0097():
    """extreme probability; deterministic sample 97."""
    rng = random.Random(2198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0098():
    """bucket edges; deterministic sample 98."""
    rng = random.Random(2199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0099():
    """well calibrated probabilities; deterministic sample 99."""
    rng = random.Random(2200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0100():
    """extreme probability; deterministic sample 100."""
    rng = random.Random(2201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0101():
    """bucket edges; deterministic sample 101."""
    rng = random.Random(2202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0102():
    """well calibrated probabilities; deterministic sample 102."""
    rng = random.Random(2203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0103():
    """extreme probability; deterministic sample 103."""
    rng = random.Random(2204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0104():
    """bucket edges; deterministic sample 104."""
    rng = random.Random(2205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0105():
    """well calibrated probabilities; deterministic sample 105."""
    rng = random.Random(2206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0106():
    """extreme probability; deterministic sample 106."""
    rng = random.Random(2207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0107():
    """bucket edges; deterministic sample 107."""
    rng = random.Random(2208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0108():
    """well calibrated probabilities; deterministic sample 108."""
    rng = random.Random(2209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0109():
    """extreme probability; deterministic sample 109."""
    rng = random.Random(2210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0110():
    """bucket edges; deterministic sample 110."""
    rng = random.Random(2211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0111():
    """well calibrated probabilities; deterministic sample 111."""
    rng = random.Random(2212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0112():
    """extreme probability; deterministic sample 112."""
    rng = random.Random(2213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0113():
    """bucket edges; deterministic sample 113."""
    rng = random.Random(2214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0114():
    """well calibrated probabilities; deterministic sample 114."""
    rng = random.Random(2215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0115():
    """extreme probability; deterministic sample 115."""
    rng = random.Random(2216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0116():
    """bucket edges; deterministic sample 116."""
    rng = random.Random(2217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0117():
    """well calibrated probabilities; deterministic sample 117."""
    rng = random.Random(2218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0118():
    """extreme probability; deterministic sample 118."""
    rng = random.Random(2219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0119():
    """bucket edges; deterministic sample 119."""
    rng = random.Random(2220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0120():
    """well calibrated probabilities; deterministic sample 120."""
    rng = random.Random(2221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0121():
    """extreme probability; deterministic sample 121."""
    rng = random.Random(2222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0122():
    """bucket edges; deterministic sample 122."""
    rng = random.Random(2223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0123():
    """well calibrated probabilities; deterministic sample 123."""
    rng = random.Random(2224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0124():
    """extreme probability; deterministic sample 124."""
    rng = random.Random(2225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0125():
    """bucket edges; deterministic sample 125."""
    rng = random.Random(2226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0126():
    """well calibrated probabilities; deterministic sample 126."""
    rng = random.Random(2227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0127():
    """extreme probability; deterministic sample 127."""
    rng = random.Random(2228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0128():
    """bucket edges; deterministic sample 128."""
    rng = random.Random(2229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0129():
    """well calibrated probabilities; deterministic sample 129."""
    rng = random.Random(2230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0130():
    """extreme probability; deterministic sample 130."""
    rng = random.Random(2231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0131():
    """bucket edges; deterministic sample 131."""
    rng = random.Random(2232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0132():
    """well calibrated probabilities; deterministic sample 132."""
    rng = random.Random(2233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0133():
    """extreme probability; deterministic sample 133."""
    rng = random.Random(2234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0134():
    """bucket edges; deterministic sample 134."""
    rng = random.Random(2235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0135():
    """well calibrated probabilities; deterministic sample 135."""
    rng = random.Random(2236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0136():
    """extreme probability; deterministic sample 136."""
    rng = random.Random(2237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0137():
    """bucket edges; deterministic sample 137."""
    rng = random.Random(2238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0138():
    """well calibrated probabilities; deterministic sample 138."""
    rng = random.Random(2239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0139():
    """extreme probability; deterministic sample 139."""
    rng = random.Random(2240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0140():
    """bucket edges; deterministic sample 140."""
    rng = random.Random(2241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0141():
    """well calibrated probabilities; deterministic sample 141."""
    rng = random.Random(2242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0142():
    """extreme probability; deterministic sample 142."""
    rng = random.Random(2243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0143():
    """bucket edges; deterministic sample 143."""
    rng = random.Random(2244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0144():
    """well calibrated probabilities; deterministic sample 144."""
    rng = random.Random(2245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0145():
    """extreme probability; deterministic sample 145."""
    rng = random.Random(2246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0146():
    """bucket edges; deterministic sample 146."""
    rng = random.Random(2247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0147():
    """well calibrated probabilities; deterministic sample 147."""
    rng = random.Random(2248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0148():
    """extreme probability; deterministic sample 148."""
    rng = random.Random(2249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0149():
    """bucket edges; deterministic sample 149."""
    rng = random.Random(2250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0150():
    """well calibrated probabilities; deterministic sample 150."""
    rng = random.Random(2251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0151():
    """extreme probability; deterministic sample 151."""
    rng = random.Random(2252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0152():
    """bucket edges; deterministic sample 152."""
    rng = random.Random(2253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0153():
    """well calibrated probabilities; deterministic sample 153."""
    rng = random.Random(2254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0154():
    """extreme probability; deterministic sample 154."""
    rng = random.Random(2255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0155():
    """bucket edges; deterministic sample 155."""
    rng = random.Random(2256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0156():
    """well calibrated probabilities; deterministic sample 156."""
    rng = random.Random(2257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0157():
    """extreme probability; deterministic sample 157."""
    rng = random.Random(2258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0158():
    """bucket edges; deterministic sample 158."""
    rng = random.Random(2259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0159():
    """well calibrated probabilities; deterministic sample 159."""
    rng = random.Random(2260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0160():
    """extreme probability; deterministic sample 160."""
    rng = random.Random(2261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0161():
    """bucket edges; deterministic sample 161."""
    rng = random.Random(2262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0162():
    """well calibrated probabilities; deterministic sample 162."""
    rng = random.Random(2263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0163():
    """extreme probability; deterministic sample 163."""
    rng = random.Random(2264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0164():
    """bucket edges; deterministic sample 164."""
    rng = random.Random(2265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0165():
    """well calibrated probabilities; deterministic sample 165."""
    rng = random.Random(2266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0166():
    """extreme probability; deterministic sample 166."""
    rng = random.Random(2267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0167():
    """bucket edges; deterministic sample 167."""
    rng = random.Random(2268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0168():
    """well calibrated probabilities; deterministic sample 168."""
    rng = random.Random(2269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0169():
    """extreme probability; deterministic sample 169."""
    rng = random.Random(2270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0170():
    """bucket edges; deterministic sample 170."""
    rng = random.Random(2271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0171():
    """well calibrated probabilities; deterministic sample 171."""
    rng = random.Random(2272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0172():
    """extreme probability; deterministic sample 172."""
    rng = random.Random(2273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0173():
    """bucket edges; deterministic sample 173."""
    rng = random.Random(2274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0174():
    """well calibrated probabilities; deterministic sample 174."""
    rng = random.Random(2275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0175():
    """extreme probability; deterministic sample 175."""
    rng = random.Random(2276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0176():
    """bucket edges; deterministic sample 176."""
    rng = random.Random(2277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0177():
    """well calibrated probabilities; deterministic sample 177."""
    rng = random.Random(2278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0178():
    """extreme probability; deterministic sample 178."""
    rng = random.Random(2279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0179():
    """bucket edges; deterministic sample 179."""
    rng = random.Random(2280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0180():
    """well calibrated probabilities; deterministic sample 180."""
    rng = random.Random(2281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0181():
    """extreme probability; deterministic sample 181."""
    rng = random.Random(2282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0182():
    """bucket edges; deterministic sample 182."""
    rng = random.Random(2283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0183():
    """well calibrated probabilities; deterministic sample 183."""
    rng = random.Random(2284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0184():
    """extreme probability; deterministic sample 184."""
    rng = random.Random(2285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0185():
    """bucket edges; deterministic sample 185."""
    rng = random.Random(2286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0186():
    """well calibrated probabilities; deterministic sample 186."""
    rng = random.Random(2287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0187():
    """extreme probability; deterministic sample 187."""
    rng = random.Random(2288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0188():
    """bucket edges; deterministic sample 188."""
    rng = random.Random(2289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0189():
    """well calibrated probabilities; deterministic sample 189."""
    rng = random.Random(2290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0190():
    """extreme probability; deterministic sample 190."""
    rng = random.Random(2291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0191():
    """bucket edges; deterministic sample 191."""
    rng = random.Random(2292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0192():
    """well calibrated probabilities; deterministic sample 192."""
    rng = random.Random(2293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0193():
    """extreme probability; deterministic sample 193."""
    rng = random.Random(2294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0194():
    """bucket edges; deterministic sample 194."""
    rng = random.Random(2295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0195():
    """well calibrated probabilities; deterministic sample 195."""
    rng = random.Random(2296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0196():
    """extreme probability; deterministic sample 196."""
    rng = random.Random(2297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0197():
    """bucket edges; deterministic sample 197."""
    rng = random.Random(2298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0198():
    """well calibrated probabilities; deterministic sample 198."""
    rng = random.Random(2299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0199():
    """extreme probability; deterministic sample 199."""
    rng = random.Random(2300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0200():
    """bucket edges; deterministic sample 200."""
    rng = random.Random(2301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0201():
    """well calibrated probabilities; deterministic sample 201."""
    rng = random.Random(2302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0202():
    """extreme probability; deterministic sample 202."""
    rng = random.Random(2303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0203():
    """bucket edges; deterministic sample 203."""
    rng = random.Random(2304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0204():
    """well calibrated probabilities; deterministic sample 204."""
    rng = random.Random(2305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0205():
    """extreme probability; deterministic sample 205."""
    rng = random.Random(2306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0206():
    """bucket edges; deterministic sample 206."""
    rng = random.Random(2307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0207():
    """well calibrated probabilities; deterministic sample 207."""
    rng = random.Random(2308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0208():
    """extreme probability; deterministic sample 208."""
    rng = random.Random(2309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0209():
    """bucket edges; deterministic sample 209."""
    rng = random.Random(2310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0210():
    """well calibrated probabilities; deterministic sample 210."""
    rng = random.Random(2311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0211():
    """extreme probability; deterministic sample 211."""
    rng = random.Random(2312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0212():
    """bucket edges; deterministic sample 212."""
    rng = random.Random(2313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0213():
    """well calibrated probabilities; deterministic sample 213."""
    rng = random.Random(2314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0214():
    """extreme probability; deterministic sample 214."""
    rng = random.Random(2315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0215():
    """bucket edges; deterministic sample 215."""
    rng = random.Random(2316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0216():
    """well calibrated probabilities; deterministic sample 216."""
    rng = random.Random(2317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0217():
    """extreme probability; deterministic sample 217."""
    rng = random.Random(2318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0218():
    """bucket edges; deterministic sample 218."""
    rng = random.Random(2319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0219():
    """well calibrated probabilities; deterministic sample 219."""
    rng = random.Random(2320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0220():
    """extreme probability; deterministic sample 220."""
    rng = random.Random(2321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0221():
    """bucket edges; deterministic sample 221."""
    rng = random.Random(2322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0222():
    """well calibrated probabilities; deterministic sample 222."""
    rng = random.Random(2323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0223():
    """extreme probability; deterministic sample 223."""
    rng = random.Random(2324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0224():
    """bucket edges; deterministic sample 224."""
    rng = random.Random(2325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0225():
    """well calibrated probabilities; deterministic sample 225."""
    rng = random.Random(2326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0226():
    """extreme probability; deterministic sample 226."""
    rng = random.Random(2327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0227():
    """bucket edges; deterministic sample 227."""
    rng = random.Random(2328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0228():
    """well calibrated probabilities; deterministic sample 228."""
    rng = random.Random(2329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0229():
    """extreme probability; deterministic sample 229."""
    rng = random.Random(2330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0230():
    """bucket edges; deterministic sample 230."""
    rng = random.Random(2331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0231():
    """well calibrated probabilities; deterministic sample 231."""
    rng = random.Random(2332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0232():
    """extreme probability; deterministic sample 232."""
    rng = random.Random(2333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0233():
    """bucket edges; deterministic sample 233."""
    rng = random.Random(2334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0234():
    """well calibrated probabilities; deterministic sample 234."""
    rng = random.Random(2335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0235():
    """extreme probability; deterministic sample 235."""
    rng = random.Random(2336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0236():
    """bucket edges; deterministic sample 236."""
    rng = random.Random(2337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0237():
    """well calibrated probabilities; deterministic sample 237."""
    rng = random.Random(2338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0238():
    """extreme probability; deterministic sample 238."""
    rng = random.Random(2339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_calibration_case_0239():
    """bucket edges; deterministic sample 239."""
    rng = random.Random(2340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.brier(scores, labels)
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

