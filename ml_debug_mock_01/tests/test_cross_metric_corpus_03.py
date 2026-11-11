"""Property-driven corpus 3: randomized, reproducible, separate test per seed."""
import random
import math
import pytest
from ml_eval.analytics import classification as c, regression as r, ranking as k, calibration as z

def test_cross_metric_corpus_3_000():
    rng=random.Random(37000)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_001():
    rng=random.Random(37001)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_002():
    rng=random.Random(37002)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_003():
    rng=random.Random(37003)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_004():
    rng=random.Random(37004)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_005():
    rng=random.Random(37005)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_006():
    rng=random.Random(37006)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_007():
    rng=random.Random(37007)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_008():
    rng=random.Random(37008)
    labels=[rng.randrange(2) for _ in range(12)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_009():
    rng=random.Random(37009)
    labels=[rng.randrange(2) for _ in range(13)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_010():
    rng=random.Random(37010)
    labels=[rng.randrange(2) for _ in range(14)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_011():
    rng=random.Random(37011)
    labels=[rng.randrange(2) for _ in range(15)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_012():
    rng=random.Random(37012)
    labels=[rng.randrange(2) for _ in range(16)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_013():
    rng=random.Random(37013)
    labels=[rng.randrange(2) for _ in range(17)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_014():
    rng=random.Random(37014)
    labels=[rng.randrange(2) for _ in range(18)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_015():
    rng=random.Random(37015)
    labels=[rng.randrange(2) for _ in range(19)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_016():
    rng=random.Random(37016)
    labels=[rng.randrange(2) for _ in range(20)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_017():
    rng=random.Random(37017)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_018():
    rng=random.Random(37018)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_019():
    rng=random.Random(37019)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_020():
    rng=random.Random(37020)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_021():
    rng=random.Random(37021)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_022():
    rng=random.Random(37022)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_023():
    rng=random.Random(37023)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_024():
    rng=random.Random(37024)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_025():
    rng=random.Random(37025)
    labels=[rng.randrange(2) for _ in range(12)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_026():
    rng=random.Random(37026)
    labels=[rng.randrange(2) for _ in range(13)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_027():
    rng=random.Random(37027)
    labels=[rng.randrange(2) for _ in range(14)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_028():
    rng=random.Random(37028)
    labels=[rng.randrange(2) for _ in range(15)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_029():
    rng=random.Random(37029)
    labels=[rng.randrange(2) for _ in range(16)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_030():
    rng=random.Random(37030)
    labels=[rng.randrange(2) for _ in range(17)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_031():
    rng=random.Random(37031)
    labels=[rng.randrange(2) for _ in range(18)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_032():
    rng=random.Random(37032)
    labels=[rng.randrange(2) for _ in range(19)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_033():
    rng=random.Random(37033)
    labels=[rng.randrange(2) for _ in range(20)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_034():
    rng=random.Random(37034)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_035():
    rng=random.Random(37035)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_036():
    rng=random.Random(37036)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_037():
    rng=random.Random(37037)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_038():
    rng=random.Random(37038)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_039():
    rng=random.Random(37039)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_040():
    rng=random.Random(37040)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_041():
    rng=random.Random(37041)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_042():
    rng=random.Random(37042)
    labels=[rng.randrange(2) for _ in range(12)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_043():
    rng=random.Random(37043)
    labels=[rng.randrange(2) for _ in range(13)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_044():
    rng=random.Random(37044)
    labels=[rng.randrange(2) for _ in range(14)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_045():
    rng=random.Random(37045)
    labels=[rng.randrange(2) for _ in range(15)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_046():
    rng=random.Random(37046)
    labels=[rng.randrange(2) for _ in range(16)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_047():
    rng=random.Random(37047)
    labels=[rng.randrange(2) for _ in range(17)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_048():
    rng=random.Random(37048)
    labels=[rng.randrange(2) for _ in range(18)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_049():
    rng=random.Random(37049)
    labels=[rng.randrange(2) for _ in range(19)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_050():
    rng=random.Random(37050)
    labels=[rng.randrange(2) for _ in range(20)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_051():
    rng=random.Random(37051)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_052():
    rng=random.Random(37052)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_053():
    rng=random.Random(37053)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_054():
    rng=random.Random(37054)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_055():
    rng=random.Random(37055)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_056():
    rng=random.Random(37056)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_057():
    rng=random.Random(37057)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_058():
    rng=random.Random(37058)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_059():
    rng=random.Random(37059)
    labels=[rng.randrange(2) for _ in range(12)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_060():
    rng=random.Random(37060)
    labels=[rng.randrange(2) for _ in range(13)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_061():
    rng=random.Random(37061)
    labels=[rng.randrange(2) for _ in range(14)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_062():
    rng=random.Random(37062)
    labels=[rng.randrange(2) for _ in range(15)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_063():
    rng=random.Random(37063)
    labels=[rng.randrange(2) for _ in range(16)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_064():
    rng=random.Random(37064)
    labels=[rng.randrange(2) for _ in range(17)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_065():
    rng=random.Random(37065)
    labels=[rng.randrange(2) for _ in range(18)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_066():
    rng=random.Random(37066)
    labels=[rng.randrange(2) for _ in range(19)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_067():
    rng=random.Random(37067)
    labels=[rng.randrange(2) for _ in range(20)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_068():
    rng=random.Random(37068)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_069():
    rng=random.Random(37069)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_070():
    rng=random.Random(37070)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_071():
    rng=random.Random(37071)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_072():
    rng=random.Random(37072)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_073():
    rng=random.Random(37073)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_074():
    rng=random.Random(37074)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_075():
    rng=random.Random(37075)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_076():
    rng=random.Random(37076)
    labels=[rng.randrange(2) for _ in range(12)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_077():
    rng=random.Random(37077)
    labels=[rng.randrange(2) for _ in range(13)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_078():
    rng=random.Random(37078)
    labels=[rng.randrange(2) for _ in range(14)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_079():
    rng=random.Random(37079)
    labels=[rng.randrange(2) for _ in range(15)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_080():
    rng=random.Random(37080)
    labels=[rng.randrange(2) for _ in range(16)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_081():
    rng=random.Random(37081)
    labels=[rng.randrange(2) for _ in range(17)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_082():
    rng=random.Random(37082)
    labels=[rng.randrange(2) for _ in range(18)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_083():
    rng=random.Random(37083)
    labels=[rng.randrange(2) for _ in range(19)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_084():
    rng=random.Random(37084)
    labels=[rng.randrange(2) for _ in range(20)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_085():
    rng=random.Random(37085)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_086():
    rng=random.Random(37086)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_087():
    rng=random.Random(37087)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_088():
    rng=random.Random(37088)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_089():
    rng=random.Random(37089)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_090():
    rng=random.Random(37090)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_091():
    rng=random.Random(37091)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_092():
    rng=random.Random(37092)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_093():
    rng=random.Random(37093)
    labels=[rng.randrange(2) for _ in range(12)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_094():
    rng=random.Random(37094)
    labels=[rng.randrange(2) for _ in range(13)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_095():
    rng=random.Random(37095)
    labels=[rng.randrange(2) for _ in range(14)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_096():
    rng=random.Random(37096)
    labels=[rng.randrange(2) for _ in range(15)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_097():
    rng=random.Random(37097)
    labels=[rng.randrange(2) for _ in range(16)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_098():
    rng=random.Random(37098)
    labels=[rng.randrange(2) for _ in range(17)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_099():
    rng=random.Random(37099)
    labels=[rng.randrange(2) for _ in range(18)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_100():
    rng=random.Random(37100)
    labels=[rng.randrange(2) for _ in range(19)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_101():
    rng=random.Random(37101)
    labels=[rng.randrange(2) for _ in range(20)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_102():
    rng=random.Random(37102)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_103():
    rng=random.Random(37103)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_104():
    rng=random.Random(37104)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_105():
    rng=random.Random(37105)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_106():
    rng=random.Random(37106)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_107():
    rng=random.Random(37107)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_108():
    rng=random.Random(37108)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_3_109():
    rng=random.Random(37109)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

