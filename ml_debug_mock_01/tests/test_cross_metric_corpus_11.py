"""Cross-metric reference corpus 11, reproducible and CPU-only."""
import random
import math
import pytest
from ml_eval.analytics import classification as c, regression as r, ranking as k, calibration as z

def test_cross_metric_corpus_11_000():
    rng=random.Random(117000)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_001():
    rng=random.Random(117001)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_002():
    rng=random.Random(117002)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_003():
    rng=random.Random(117003)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_004():
    rng=random.Random(117004)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_005():
    rng=random.Random(117005)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_006():
    rng=random.Random(117006)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_007():
    rng=random.Random(117007)
    labels=[rng.randrange(2) for _ in range(11)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_008():
    rng=random.Random(117008)
    labels=[rng.randrange(2) for _ in range(12)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_009():
    rng=random.Random(117009)
    labels=[rng.randrange(2) for _ in range(13)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_010():
    rng=random.Random(117010)
    labels=[rng.randrange(2) for _ in range(14)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_011():
    rng=random.Random(117011)
    labels=[rng.randrange(2) for _ in range(15)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_012():
    rng=random.Random(117012)
    labels=[rng.randrange(2) for _ in range(16)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_013():
    rng=random.Random(117013)
    labels=[rng.randrange(2) for _ in range(17)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_014():
    rng=random.Random(117014)
    labels=[rng.randrange(2) for _ in range(18)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_015():
    rng=random.Random(117015)
    labels=[rng.randrange(2) for _ in range(19)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_016():
    rng=random.Random(117016)
    labels=[rng.randrange(2) for _ in range(20)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_017():
    rng=random.Random(117017)
    labels=[rng.randrange(2) for _ in range(4)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_018():
    rng=random.Random(117018)
    labels=[rng.randrange(2) for _ in range(5)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_019():
    rng=random.Random(117019)
    labels=[rng.randrange(2) for _ in range(6)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_020():
    rng=random.Random(117020)
    labels=[rng.randrange(2) for _ in range(7)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_021():
    rng=random.Random(117021)
    labels=[rng.randrange(2) for _ in range(8)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_022():
    rng=random.Random(117022)
    labels=[rng.randrange(2) for _ in range(9)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

def test_cross_metric_corpus_11_023():
    rng=random.Random(117023)
    labels=[rng.randrange(2) for _ in range(10)]
    scores=[rng.random() for _ in labels]
    predicted=c.threshold_scores(scores,0.5)
    assert 0 <= c.accuracy(predicted,labels) <= 1
    assert 0 <= c.fbeta(predicted,labels) <= 1
    assert 0 <= z.brier(scores,labels) <= 1
    assert r.mse(scores,labels) == pytest.approx(z.brier(scores,labels))
    assert 0 <= k.ndcg(scores,labels) <= 1.000000001

