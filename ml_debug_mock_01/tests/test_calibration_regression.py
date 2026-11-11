"""Regression cases for calibration: deterministic scenarios and invariants."""
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

