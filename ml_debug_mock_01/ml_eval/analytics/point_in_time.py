"""Point-in-time feature generation for event-based recommendation systems.

Online inference must not observe feature values that arrived after the request.
Timestamps are assumed comparable, ideally timezone-aware datetimes.
"""
from __future__ import annotations
from collections import defaultdict


def join_latest_features(impressions, feature_events, *,
                         entity_key="user_id", impression_time="event_time",
                         feature_time="event_time", arrival_time="ingested_at"):
    """Attach most recent historically available feature for each entity.

    Eligibility requires feature_time <= impression_time AND arrival_time <=
    impression_time. Return results in original impression order, unmatched None.
    """
    index=defaultdict(list)
    for feat in feature_events:
        index[feat[entity_key]].append(feat)
    for values in index.values():
        values.sort(key=lambda row:row[feature_time])
    results=[]
    for event in impressions:
        key=event[entity_key]
        ts=event[impression_time]
        match=None
        for candidate in index.get(key,()):
            if candidate[feature_time]<=ts:
                match=candidate
            else:
                break
        results.append({"impression":event,"feature":match})
    return results


def rolling_user_ctr(events, *, entity_key="user_id", time_key="event_time",
                     label_key="clicked", window=20):
    """CTR over prior observations only, excluding current event.

    Equal timestamps use stable original input order as the tie-breaker.
    Return output in original input order.
    """
    if window<=0:raise ValueError("window must be positive")
    ordered=sorted(enumerate(events),key=lambda p:(p[1][time_key],p[0]))
    history=defaultdict(list)
    result=[None]*len(events)
    for index,event in ordered:
        user=event[entity_key]
        series=history[user]
        series.append(int(event[label_key]))
        recent=series[-window:]
        result[index]=sum(recent)/len(recent) if recent else 0.0
    return result


def time_based_train_test(events, cutoff, *, time_key="event_time",
                          label_time_key="label_available_at"):
    """Train only on pre-cutoff events whose labels are available by cutoff.

    Events at/after cutoff are the test population; old examples with
    unavailable labels are neither train nor test.
    """
    train=[]
    test=[]
    for row in events:
        if row[time_key]<cutoff:
            train.append(row)
        else:
            test.append(row)
    return train,test
