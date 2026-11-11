"""ML feature consistency, leakage, late arrival and reproducibility contracts."""
from datetime import datetime,timedelta
import random
import pytest
from ml_eval.analytics.point_in_time import (
    join_latest_features,rolling_user_ctr,time_based_train_test
)
BASE=datetime(2026,1,1)
def dt(day):return BASE+timedelta(days=day)


def test_asof_join_must_respect_ingestion_timestamp():
    impressions=[{"user_id":"u1","event_time":dt(5)},
                 {"user_id":"u1","event_time":dt(10)}]
    features=[
      {"user_id":"u1","event_time":dt(1),"ingested_at":dt(2),"value":0.2},
      {"user_id":"u1","event_time":dt(4),"ingested_at":dt(7),"value":0.9},
      {"user_id":"u1","event_time":dt(8),"ingested_at":dt(9),"value":0.6}]
    joined=join_latest_features(impressions,features)
    assert [r["feature"]["value"] for r in joined]==[0.2,0.6]


def test_asof_join_preserves_impression_order_with_missing_entities():
    impressions=[{"user_id":"missing","event_time":dt(10)},
                 {"user_id":"u1","event_time":dt(3)},
                 {"user_id":"u1","event_time":dt(1)}]
    features=[{"user_id":"u1","event_time":dt(2),"ingested_at":dt(2),"value":1}]
    joined=join_latest_features(impressions,features)
    assert [x["feature"]["value"] if x["feature"] else None for x in joined]==[None,1,None]


def test_join_does_not_mutate_feature_rows():
    features=[{"user_id":"u1","event_time":dt(2),"ingested_at":dt(3),"value":10},
              {"user_id":"u1","event_time":dt(1),"ingested_at":dt(1),"value":9}]
    original=[dict(r) for r in features]
    join_latest_features([{"user_id":"u1","event_time":dt(5)}],features)
    assert features==original


def test_rolling_ctr_excludes_current_example():
    events=[{"user_id":"a","event_time":dt(i),"clicked":v}
            for i,v in enumerate([1,0,0,1])]
    assert rolling_user_ctr(events,window=2)==pytest.approx([0.0,1.0,0.5,0.0])


def test_rolling_ctr_does_not_mix_entities():
    events=[{"user_id":"a","event_time":dt(1),"clicked":1},
            {"user_id":"b","event_time":dt(2),"clicked":0},
            {"user_id":"a","event_time":dt(3),"clicked":0},
            {"user_id":"b","event_time":dt(4),"clicked":1}]
    assert rolling_user_ctr(events,window=2)==pytest.approx([0.0,0.0,1.0,0.0])


def test_training_excludes_labels_arriving_after_cutoff():
    rows=[{"id":"ok","event_time":dt(1),"label_available_at":dt(2)},
          {"id":"delayed","event_time":dt(3),"label_available_at":dt(9)},
          {"id":"boundary","event_time":dt(5),"label_available_at":dt(8)}]
    train,test=time_based_train_test(rows,dt(5))
    assert [r["id"] for r in train]==["ok"]
    assert [r["id"] for r in test]==["boundary"]


@pytest.mark.parametrize("seed",[2,7,19])
def test_asof_join_matches_bruteforce_reference(seed):
    rng=random.Random(seed)
    features=[
        {"user_id":f"u{rng.randrange(4)}","event_time":dt(rng.randrange(1,20)),
         "ingested_at":dt(rng.randrange(1,22)),"value":i}
        for i in range(80)]
    impressions=[
        {"user_id":f"u{rng.randrange(4)}","event_time":dt(rng.randrange(1,22))}
        for _ in range(40)]
    actual=join_latest_features(impressions,features)
    for impression, row in zip(impressions,actual):
        eligible=[(i,r) for i,r in enumerate(features)
                  if r["user_id"]==impression["user_id"]
                  and r["event_time"]<=impression["event_time"]
                  and r["ingested_at"]<=impression["event_time"]]
        if not eligible:
            assert row["feature"] is None
        else:
            _,expected=max(eligible,key=lambda p:(p[1]["event_time"],p[0]))
            assert row["feature"]==expected
