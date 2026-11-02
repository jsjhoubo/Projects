from datetime import datetime, timedelta, timezone
from recordflow.domain import BusinessRecord
from recordflow.feature_export import TrainingExample, build_training_examples

U=timezone.utc

def t(day, hour=0):
    return datetime(2026,1,day,hour,tzinfo=U)

def r(id,city,when,source='atlas'):
    return BusinessRecord(id,'Name',city,when,source)

def test_example_and_future_leakage():
    snapshots=[r('a','Boston',t(3),'beacon'),r('a','NYC',t(1)),r('b','Philly',t(2))]
    labels=[('a',t(2,12)),('b',t(1,12)),('a',t(4))]
    x=build_training_examples(snapshots, labels)
    assert [(z.business_id,z.city,z.source,z.record_age_hours) for z in x]==[
        ('a','NYC','atlas',36.0),('b',None,None,None),('a','Boston','beacon',24.0)]


def test_chooses_latest_eligible_not_latest_overall():
    snapshots=[r('x','v3',t(4)),r('x','v1',t(1)),r('x','v2',t(2))]
    x=build_training_examples(snapshots,[('x',t(1)),('x',t(3)),('x',t(4))])
    assert [z.city for z in x]==['v1','v2','v3']
    assert [z.record_age_hours for z in x]==[0.0,24.0,0.0]

def test_last_equal_timestamp_wins():
    snapshots=[r('a','one',t(1)),r('a','other',t(1)),r('a','later',t(3))]
    x=build_training_examples(snapshots,[('a',t(2)),('a',t(1))])
    assert [z.city for z in x]==['other','other']

def test_duplicate_labels_and_missing_business():
    snapshots=[r('a','NYC',t(2))]
    labels=[('a',t(1)),('missing',t(5)),('a',t(3)),('a',t(3))]
    x=build_training_examples(iter(snapshots),iter(labels))
    assert len(x)==4
    assert [z.city for z in x]==[None,None,'NYC','NYC']
    assert [z.business_id for z in x]==['a','missing','a','a']

def test_hour_precision_and_large_input():
    snapshots=[r('x','old',t(1)),r('x','new',t(2))]
    at=t(2)+timedelta(minutes=30)
    assert build_training_examples(snapshots,[('x',at)])[0].record_age_hours==0.5
    # Performance sanity check: an O(S*L) algorithm is impractical at this size.
    many=[r(f'b{i}', 'A',t(1)) for i in range(3000)]
    labels=[(f'b{i}',t(2)) for i in range(3000)]
    ans=build_training_examples(many,labels)
    assert len(ans)==3000
    assert ans[1234].record_age_hours==24.0
