"""Cross-metric reference corpus 1, reproducible and CPU-only."""
import random
import math
import pytest
from ml_eval.analytics import classification as c, regression as r, ranking as k, calibration as z

def test_cross_metric_corpus_1_000():
    rng=random.Random(17000)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_001():
    rng=random.Random(17001)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_002():
    rng=random.Random(17002)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_003():
    rng=random.Random(17003)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_004():
    rng=random.Random(17004)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_005():
    rng=random.Random(17005)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_006():
    rng=random.Random(17006)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_007():
    rng=random.Random(17007)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_008():
    rng=random.Random(17008)
    labels=[rng.randrange(2) for _ in range(12)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_009():
    rng=random.Random(17009)
    labels=[rng.randrange(2) for _ in range(13)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_010():
    rng=random.Random(17010)
    labels=[rng.randrange(2) for _ in range(14)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_011():
    rng=random.Random(17011)
    labels=[rng.randrange(2) for _ in range(15)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_012():
    rng=random.Random(17012)
    labels=[rng.randrange(2) for _ in range(16)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_013():
    rng=random.Random(17013)
    labels=[rng.randrange(2) for _ in range(17)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_014():
    rng=random.Random(17014)
    labels=[rng.randrange(2) for _ in range(18)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_015():
    rng=random.Random(17015)
    labels=[rng.randrange(2) for _ in range(19)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_016():
    rng=random.Random(17016)
    labels=[rng.randrange(2) for _ in range(20)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_017():
    rng=random.Random(17017)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_018():
    rng=random.Random(17018)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_019():
    rng=random.Random(17019)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_020():
    rng=random.Random(17020)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_021():
    rng=random.Random(17021)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_022():
    rng=random.Random(17022)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_1_023():
    rng=random.Random(17023)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

