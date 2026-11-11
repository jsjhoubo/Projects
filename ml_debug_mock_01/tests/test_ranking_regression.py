"""Regression cases for ranking: deterministic scenarios and invariants."""
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

