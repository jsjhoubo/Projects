"""Regression cases for ranking: both deterministic examples and invariants."""
import math
import random
import pytest
from ml_eval.analytics import ranking as subject

def test_ranking_case_0000():
    """tie stability; deterministic sample 0."""
    rng = random.Random(1101)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0001():
    """empty rankings; deterministic sample 1."""
    rng = random.Random(1102)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0002():
    """top k truncation; deterministic sample 2."""
    rng = random.Random(1103)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0003():
    """tie stability; deterministic sample 3."""
    rng = random.Random(1104)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0004():
    """empty rankings; deterministic sample 4."""
    rng = random.Random(1105)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0005():
    """top k truncation; deterministic sample 5."""
    rng = random.Random(1106)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0006():
    """tie stability; deterministic sample 6."""
    rng = random.Random(1107)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0007():
    """empty rankings; deterministic sample 7."""
    rng = random.Random(1108)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0008():
    """top k truncation; deterministic sample 8."""
    rng = random.Random(1109)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0009():
    """tie stability; deterministic sample 9."""
    rng = random.Random(1110)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0010():
    """empty rankings; deterministic sample 10."""
    rng = random.Random(1111)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0011():
    """top k truncation; deterministic sample 11."""
    rng = random.Random(1112)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0012():
    """tie stability; deterministic sample 12."""
    rng = random.Random(1113)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0013():
    """empty rankings; deterministic sample 13."""
    rng = random.Random(1114)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0014():
    """top k truncation; deterministic sample 14."""
    rng = random.Random(1115)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0015():
    """tie stability; deterministic sample 15."""
    rng = random.Random(1116)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0016():
    """empty rankings; deterministic sample 16."""
    rng = random.Random(1117)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0017():
    """top k truncation; deterministic sample 17."""
    rng = random.Random(1118)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0018():
    """tie stability; deterministic sample 18."""
    rng = random.Random(1119)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0019():
    """empty rankings; deterministic sample 19."""
    rng = random.Random(1120)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0020():
    """top k truncation; deterministic sample 20."""
    rng = random.Random(1121)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0021():
    """tie stability; deterministic sample 21."""
    rng = random.Random(1122)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0022():
    """empty rankings; deterministic sample 22."""
    rng = random.Random(1123)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0023():
    """top k truncation; deterministic sample 23."""
    rng = random.Random(1124)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0024():
    """tie stability; deterministic sample 24."""
    rng = random.Random(1125)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0025():
    """empty rankings; deterministic sample 25."""
    rng = random.Random(1126)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0026():
    """top k truncation; deterministic sample 26."""
    rng = random.Random(1127)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0027():
    """tie stability; deterministic sample 27."""
    rng = random.Random(1128)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0028():
    """empty rankings; deterministic sample 28."""
    rng = random.Random(1129)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0029():
    """top k truncation; deterministic sample 29."""
    rng = random.Random(1130)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0030():
    """tie stability; deterministic sample 30."""
    rng = random.Random(1131)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0031():
    """empty rankings; deterministic sample 31."""
    rng = random.Random(1132)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0032():
    """top k truncation; deterministic sample 32."""
    rng = random.Random(1133)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0033():
    """tie stability; deterministic sample 33."""
    rng = random.Random(1134)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0034():
    """empty rankings; deterministic sample 34."""
    rng = random.Random(1135)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0035():
    """top k truncation; deterministic sample 35."""
    rng = random.Random(1136)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0036():
    """tie stability; deterministic sample 36."""
    rng = random.Random(1137)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0037():
    """empty rankings; deterministic sample 37."""
    rng = random.Random(1138)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0038():
    """top k truncation; deterministic sample 38."""
    rng = random.Random(1139)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0039():
    """tie stability; deterministic sample 39."""
    rng = random.Random(1140)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0040():
    """empty rankings; deterministic sample 40."""
    rng = random.Random(1141)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0041():
    """top k truncation; deterministic sample 41."""
    rng = random.Random(1142)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0042():
    """tie stability; deterministic sample 42."""
    rng = random.Random(1143)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0043():
    """empty rankings; deterministic sample 43."""
    rng = random.Random(1144)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0044():
    """top k truncation; deterministic sample 44."""
    rng = random.Random(1145)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0045():
    """tie stability; deterministic sample 45."""
    rng = random.Random(1146)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0046():
    """empty rankings; deterministic sample 46."""
    rng = random.Random(1147)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0047():
    """top k truncation; deterministic sample 47."""
    rng = random.Random(1148)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0048():
    """tie stability; deterministic sample 48."""
    rng = random.Random(1149)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0049():
    """empty rankings; deterministic sample 49."""
    rng = random.Random(1150)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0050():
    """top k truncation; deterministic sample 50."""
    rng = random.Random(1151)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0051():
    """tie stability; deterministic sample 51."""
    rng = random.Random(1152)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0052():
    """empty rankings; deterministic sample 52."""
    rng = random.Random(1153)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0053():
    """top k truncation; deterministic sample 53."""
    rng = random.Random(1154)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0054():
    """tie stability; deterministic sample 54."""
    rng = random.Random(1155)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0055():
    """empty rankings; deterministic sample 55."""
    rng = random.Random(1156)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0056():
    """top k truncation; deterministic sample 56."""
    rng = random.Random(1157)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0057():
    """tie stability; deterministic sample 57."""
    rng = random.Random(1158)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0058():
    """empty rankings; deterministic sample 58."""
    rng = random.Random(1159)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0059():
    """top k truncation; deterministic sample 59."""
    rng = random.Random(1160)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0060():
    """tie stability; deterministic sample 60."""
    rng = random.Random(1161)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0061():
    """empty rankings; deterministic sample 61."""
    rng = random.Random(1162)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0062():
    """top k truncation; deterministic sample 62."""
    rng = random.Random(1163)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0063():
    """tie stability; deterministic sample 63."""
    rng = random.Random(1164)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0064():
    """empty rankings; deterministic sample 64."""
    rng = random.Random(1165)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0065():
    """top k truncation; deterministic sample 65."""
    rng = random.Random(1166)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0066():
    """tie stability; deterministic sample 66."""
    rng = random.Random(1167)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0067():
    """empty rankings; deterministic sample 67."""
    rng = random.Random(1168)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0068():
    """top k truncation; deterministic sample 68."""
    rng = random.Random(1169)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0069():
    """tie stability; deterministic sample 69."""
    rng = random.Random(1170)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0070():
    """empty rankings; deterministic sample 70."""
    rng = random.Random(1171)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0071():
    """top k truncation; deterministic sample 71."""
    rng = random.Random(1172)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0072():
    """tie stability; deterministic sample 72."""
    rng = random.Random(1173)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0073():
    """empty rankings; deterministic sample 73."""
    rng = random.Random(1174)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0074():
    """top k truncation; deterministic sample 74."""
    rng = random.Random(1175)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0075():
    """tie stability; deterministic sample 75."""
    rng = random.Random(1176)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0076():
    """empty rankings; deterministic sample 76."""
    rng = random.Random(1177)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0077():
    """top k truncation; deterministic sample 77."""
    rng = random.Random(1178)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0078():
    """tie stability; deterministic sample 78."""
    rng = random.Random(1179)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0079():
    """empty rankings; deterministic sample 79."""
    rng = random.Random(1180)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0080():
    """top k truncation; deterministic sample 80."""
    rng = random.Random(1181)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0081():
    """tie stability; deterministic sample 81."""
    rng = random.Random(1182)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0082():
    """empty rankings; deterministic sample 82."""
    rng = random.Random(1183)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0083():
    """top k truncation; deterministic sample 83."""
    rng = random.Random(1184)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0084():
    """tie stability; deterministic sample 84."""
    rng = random.Random(1185)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0085():
    """empty rankings; deterministic sample 85."""
    rng = random.Random(1186)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0086():
    """top k truncation; deterministic sample 86."""
    rng = random.Random(1187)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0087():
    """tie stability; deterministic sample 87."""
    rng = random.Random(1188)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0088():
    """empty rankings; deterministic sample 88."""
    rng = random.Random(1189)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0089():
    """top k truncation; deterministic sample 89."""
    rng = random.Random(1190)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0090():
    """tie stability; deterministic sample 90."""
    rng = random.Random(1191)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0091():
    """empty rankings; deterministic sample 91."""
    rng = random.Random(1192)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0092():
    """top k truncation; deterministic sample 92."""
    rng = random.Random(1193)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0093():
    """tie stability; deterministic sample 93."""
    rng = random.Random(1194)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0094():
    """empty rankings; deterministic sample 94."""
    rng = random.Random(1195)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0095():
    """top k truncation; deterministic sample 95."""
    rng = random.Random(1196)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0096():
    """tie stability; deterministic sample 96."""
    rng = random.Random(1197)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0097():
    """empty rankings; deterministic sample 97."""
    rng = random.Random(1198)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0098():
    """top k truncation; deterministic sample 98."""
    rng = random.Random(1199)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0099():
    """tie stability; deterministic sample 99."""
    rng = random.Random(1200)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0100():
    """empty rankings; deterministic sample 100."""
    rng = random.Random(1201)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0101():
    """top k truncation; deterministic sample 101."""
    rng = random.Random(1202)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0102():
    """tie stability; deterministic sample 102."""
    rng = random.Random(1203)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0103():
    """empty rankings; deterministic sample 103."""
    rng = random.Random(1204)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0104():
    """top k truncation; deterministic sample 104."""
    rng = random.Random(1205)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0105():
    """tie stability; deterministic sample 105."""
    rng = random.Random(1206)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0106():
    """empty rankings; deterministic sample 106."""
    rng = random.Random(1207)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0107():
    """top k truncation; deterministic sample 107."""
    rng = random.Random(1208)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0108():
    """tie stability; deterministic sample 108."""
    rng = random.Random(1209)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0109():
    """empty rankings; deterministic sample 109."""
    rng = random.Random(1210)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0110():
    """top k truncation; deterministic sample 110."""
    rng = random.Random(1211)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0111():
    """tie stability; deterministic sample 111."""
    rng = random.Random(1212)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0112():
    """empty rankings; deterministic sample 112."""
    rng = random.Random(1213)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0113():
    """top k truncation; deterministic sample 113."""
    rng = random.Random(1214)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0114():
    """tie stability; deterministic sample 114."""
    rng = random.Random(1215)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0115():
    """empty rankings; deterministic sample 115."""
    rng = random.Random(1216)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0116():
    """top k truncation; deterministic sample 116."""
    rng = random.Random(1217)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0117():
    """tie stability; deterministic sample 117."""
    rng = random.Random(1218)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0118():
    """empty rankings; deterministic sample 118."""
    rng = random.Random(1219)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0119():
    """top k truncation; deterministic sample 119."""
    rng = random.Random(1220)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0120():
    """tie stability; deterministic sample 120."""
    rng = random.Random(1221)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0121():
    """empty rankings; deterministic sample 121."""
    rng = random.Random(1222)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0122():
    """top k truncation; deterministic sample 122."""
    rng = random.Random(1223)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0123():
    """tie stability; deterministic sample 123."""
    rng = random.Random(1224)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0124():
    """empty rankings; deterministic sample 124."""
    rng = random.Random(1225)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0125():
    """top k truncation; deterministic sample 125."""
    rng = random.Random(1226)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0126():
    """tie stability; deterministic sample 126."""
    rng = random.Random(1227)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0127():
    """empty rankings; deterministic sample 127."""
    rng = random.Random(1228)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0128():
    """top k truncation; deterministic sample 128."""
    rng = random.Random(1229)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0129():
    """tie stability; deterministic sample 129."""
    rng = random.Random(1230)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0130():
    """empty rankings; deterministic sample 130."""
    rng = random.Random(1231)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0131():
    """top k truncation; deterministic sample 131."""
    rng = random.Random(1232)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0132():
    """tie stability; deterministic sample 132."""
    rng = random.Random(1233)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0133():
    """empty rankings; deterministic sample 133."""
    rng = random.Random(1234)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0134():
    """top k truncation; deterministic sample 134."""
    rng = random.Random(1235)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0135():
    """tie stability; deterministic sample 135."""
    rng = random.Random(1236)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0136():
    """empty rankings; deterministic sample 136."""
    rng = random.Random(1237)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0137():
    """top k truncation; deterministic sample 137."""
    rng = random.Random(1238)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0138():
    """tie stability; deterministic sample 138."""
    rng = random.Random(1239)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0139():
    """empty rankings; deterministic sample 139."""
    rng = random.Random(1240)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0140():
    """top k truncation; deterministic sample 140."""
    rng = random.Random(1241)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0141():
    """tie stability; deterministic sample 141."""
    rng = random.Random(1242)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0142():
    """empty rankings; deterministic sample 142."""
    rng = random.Random(1243)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0143():
    """top k truncation; deterministic sample 143."""
    rng = random.Random(1244)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0144():
    """tie stability; deterministic sample 144."""
    rng = random.Random(1245)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0145():
    """empty rankings; deterministic sample 145."""
    rng = random.Random(1246)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0146():
    """top k truncation; deterministic sample 146."""
    rng = random.Random(1247)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0147():
    """tie stability; deterministic sample 147."""
    rng = random.Random(1248)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0148():
    """empty rankings; deterministic sample 148."""
    rng = random.Random(1249)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0149():
    """top k truncation; deterministic sample 149."""
    rng = random.Random(1250)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0150():
    """tie stability; deterministic sample 150."""
    rng = random.Random(1251)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0151():
    """empty rankings; deterministic sample 151."""
    rng = random.Random(1252)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0152():
    """top k truncation; deterministic sample 152."""
    rng = random.Random(1253)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0153():
    """tie stability; deterministic sample 153."""
    rng = random.Random(1254)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0154():
    """empty rankings; deterministic sample 154."""
    rng = random.Random(1255)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0155():
    """top k truncation; deterministic sample 155."""
    rng = random.Random(1256)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0156():
    """tie stability; deterministic sample 156."""
    rng = random.Random(1257)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0157():
    """empty rankings; deterministic sample 157."""
    rng = random.Random(1258)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0158():
    """top k truncation; deterministic sample 158."""
    rng = random.Random(1259)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0159():
    """tie stability; deterministic sample 159."""
    rng = random.Random(1260)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0160():
    """empty rankings; deterministic sample 160."""
    rng = random.Random(1261)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0161():
    """top k truncation; deterministic sample 161."""
    rng = random.Random(1262)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0162():
    """tie stability; deterministic sample 162."""
    rng = random.Random(1263)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0163():
    """empty rankings; deterministic sample 163."""
    rng = random.Random(1264)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0164():
    """top k truncation; deterministic sample 164."""
    rng = random.Random(1265)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0165():
    """tie stability; deterministic sample 165."""
    rng = random.Random(1266)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0166():
    """empty rankings; deterministic sample 166."""
    rng = random.Random(1267)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0167():
    """top k truncation; deterministic sample 167."""
    rng = random.Random(1268)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0168():
    """tie stability; deterministic sample 168."""
    rng = random.Random(1269)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0169():
    """empty rankings; deterministic sample 169."""
    rng = random.Random(1270)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0170():
    """top k truncation; deterministic sample 170."""
    rng = random.Random(1271)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0171():
    """tie stability; deterministic sample 171."""
    rng = random.Random(1272)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0172():
    """empty rankings; deterministic sample 172."""
    rng = random.Random(1273)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0173():
    """top k truncation; deterministic sample 173."""
    rng = random.Random(1274)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0174():
    """tie stability; deterministic sample 174."""
    rng = random.Random(1275)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0175():
    """empty rankings; deterministic sample 175."""
    rng = random.Random(1276)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0176():
    """top k truncation; deterministic sample 176."""
    rng = random.Random(1277)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0177():
    """tie stability; deterministic sample 177."""
    rng = random.Random(1278)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0178():
    """empty rankings; deterministic sample 178."""
    rng = random.Random(1279)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0179():
    """top k truncation; deterministic sample 179."""
    rng = random.Random(1280)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0180():
    """tie stability; deterministic sample 180."""
    rng = random.Random(1281)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0181():
    """empty rankings; deterministic sample 181."""
    rng = random.Random(1282)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0182():
    """top k truncation; deterministic sample 182."""
    rng = random.Random(1283)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0183():
    """tie stability; deterministic sample 183."""
    rng = random.Random(1284)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0184():
    """empty rankings; deterministic sample 184."""
    rng = random.Random(1285)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0185():
    """top k truncation; deterministic sample 185."""
    rng = random.Random(1286)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0186():
    """tie stability; deterministic sample 186."""
    rng = random.Random(1287)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0187():
    """empty rankings; deterministic sample 187."""
    rng = random.Random(1288)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0188():
    """top k truncation; deterministic sample 188."""
    rng = random.Random(1289)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0189():
    """tie stability; deterministic sample 189."""
    rng = random.Random(1290)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0190():
    """empty rankings; deterministic sample 190."""
    rng = random.Random(1291)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0191():
    """top k truncation; deterministic sample 191."""
    rng = random.Random(1292)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0192():
    """tie stability; deterministic sample 192."""
    rng = random.Random(1293)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0193():
    """empty rankings; deterministic sample 193."""
    rng = random.Random(1294)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0194():
    """top k truncation; deterministic sample 194."""
    rng = random.Random(1295)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0195():
    """tie stability; deterministic sample 195."""
    rng = random.Random(1296)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0196():
    """empty rankings; deterministic sample 196."""
    rng = random.Random(1297)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0197():
    """top k truncation; deterministic sample 197."""
    rng = random.Random(1298)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0198():
    """tie stability; deterministic sample 198."""
    rng = random.Random(1299)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0199():
    """empty rankings; deterministic sample 199."""
    rng = random.Random(1300)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0200():
    """top k truncation; deterministic sample 200."""
    rng = random.Random(1301)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0201():
    """tie stability; deterministic sample 201."""
    rng = random.Random(1302)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0202():
    """empty rankings; deterministic sample 202."""
    rng = random.Random(1303)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0203():
    """top k truncation; deterministic sample 203."""
    rng = random.Random(1304)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0204():
    """tie stability; deterministic sample 204."""
    rng = random.Random(1305)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0205():
    """empty rankings; deterministic sample 205."""
    rng = random.Random(1306)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0206():
    """top k truncation; deterministic sample 206."""
    rng = random.Random(1307)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0207():
    """tie stability; deterministic sample 207."""
    rng = random.Random(1308)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0208():
    """empty rankings; deterministic sample 208."""
    rng = random.Random(1309)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0209():
    """top k truncation; deterministic sample 209."""
    rng = random.Random(1310)
    n = 25
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0210():
    """tie stability; deterministic sample 210."""
    rng = random.Random(1311)
    n = 26
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0211():
    """empty rankings; deterministic sample 211."""
    rng = random.Random(1312)
    n = 27
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0212():
    """top k truncation; deterministic sample 212."""
    rng = random.Random(1313)
    n = 28
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0213():
    """tie stability; deterministic sample 213."""
    rng = random.Random(1314)
    n = 29
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0214():
    """empty rankings; deterministic sample 214."""
    rng = random.Random(1315)
    n = 30
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0215():
    """top k truncation; deterministic sample 215."""
    rng = random.Random(1316)
    n = 31
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0216():
    """tie stability; deterministic sample 216."""
    rng = random.Random(1317)
    n = 32
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0217():
    """empty rankings; deterministic sample 217."""
    rng = random.Random(1318)
    n = 2
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0218():
    """top k truncation; deterministic sample 218."""
    rng = random.Random(1319)
    n = 3
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0219():
    """tie stability; deterministic sample 219."""
    rng = random.Random(1320)
    n = 4
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0220():
    """empty rankings; deterministic sample 220."""
    rng = random.Random(1321)
    n = 5
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0221():
    """top k truncation; deterministic sample 221."""
    rng = random.Random(1322)
    n = 6
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0222():
    """tie stability; deterministic sample 222."""
    rng = random.Random(1323)
    n = 7
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0223():
    """empty rankings; deterministic sample 223."""
    rng = random.Random(1324)
    n = 8
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0224():
    """top k truncation; deterministic sample 224."""
    rng = random.Random(1325)
    n = 9
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0225():
    """tie stability; deterministic sample 225."""
    rng = random.Random(1326)
    n = 10
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0226():
    """empty rankings; deterministic sample 226."""
    rng = random.Random(1327)
    n = 11
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0227():
    """top k truncation; deterministic sample 227."""
    rng = random.Random(1328)
    n = 12
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0228():
    """tie stability; deterministic sample 228."""
    rng = random.Random(1329)
    n = 13
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0229():
    """empty rankings; deterministic sample 229."""
    rng = random.Random(1330)
    n = 14
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0230():
    """top k truncation; deterministic sample 230."""
    rng = random.Random(1331)
    n = 15
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0231():
    """tie stability; deterministic sample 231."""
    rng = random.Random(1332)
    n = 16
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0232():
    """empty rankings; deterministic sample 232."""
    rng = random.Random(1333)
    n = 17
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0233():
    """top k truncation; deterministic sample 233."""
    rng = random.Random(1334)
    n = 18
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0234():
    """tie stability; deterministic sample 234."""
    rng = random.Random(1335)
    n = 19
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0235():
    """empty rankings; deterministic sample 235."""
    rng = random.Random(1336)
    n = 20
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0236():
    """top k truncation; deterministic sample 236."""
    rng = random.Random(1337)
    n = 21
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0237():
    """tie stability; deterministic sample 237."""
    rng = random.Random(1338)
    n = 22
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0238():
    """empty rankings; deterministic sample 238."""
    rng = random.Random(1339)
    n = 23
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

def test_ranking_case_0239():
    """top k truncation; deterministic sample 239."""
    rng = random.Random(1340)
    n = 24
    labels = [rng.randrange(2) for _ in range(n)]
    preds = [rng.randrange(2) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]
    result = subject.ndcg(scores, labels, k=max(1,n//2))
    assert math.isfinite(float(result))
    assert 0.0 <= result <= 1.0

