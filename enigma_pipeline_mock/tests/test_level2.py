from datetime import datetime, timezone
from recordflow.ingestion import IngestionService
from recordflow.domain import BusinessRecord, FeedEnvelope
from recordflow.storage import MemoryBusinessStore
from recordflow.audit import MemoryAuditLog
from recordflow.metrics import IngestCounters


def setup():
    store = MemoryBusinessStore([BusinessRecord('a', 'Old Name', 'NYC', datetime(2026, 1, 1, tzinfo=timezone.utc), 'atlas')])
    audit = MemoryAuditLog()
    counters = IngestCounters()
    return IngestionService(store, audit, counters), store, audit, counters


def atlas(id='a', ts='2026-01-02T00:00:00Z', **fields):
    return FeedEnvelope(' ATLAS ', {'business_id': id, 'updated_at': ts, **fields})


def test_example_merge_stale_missing():
    svc, store, audit, counters = setup()
    r = svc.ingest_patches([
        atlas(' A ', name='New Name'),
        atlas('a', '2026-01-01T12:00:00Z', city='Boston'),
        atlas('missing', '2026-01-03T00:00:00Z', name='X'),
    ])
    assert (r.inserted, r.updated, r.stale, r.rejected) == (0, 1, 1, 1)
    assert [e.action for e in r.events] == ['updated','stale','rejected']
    assert [e.business_id for e in r.events] == ['a','a',None]
    assert all(e.supplier == 'atlas' for e in r.events)
    assert store.get('a').name == 'New Name' and store.get('a').city == 'NYC'
    assert store.get('a').updated_at == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert store.get('missing') is None
    assert audit.all_events() == list(r.events)
    assert (counters.batches, counters.updated, counters.stale, counters.rejected) == (1,1,1,1)


def test_beacon_partial_city_and_timezone_conversion():
    svc, store, _, _ = setup()
    r = svc.ingest_patches([FeedEnvelope('BEACON', {
        'entityId':'A', 'timestamp':'2026-01-02T10:00:00+02:00', 'location':{'city':'  Boston  '}
    })])
    assert (r.updated, r.rejected) == (1,0)
    item = store.get('a')
    assert (item.name, item.city, item.source) == ('Old Name','Boston','beacon')
    assert item.updated_at == datetime(2026,1,2,8,tzinfo=timezone.utc)
    assert r.events[0].supplier == 'beacon'


def test_invalid_fields_do_not_partially_mutate():
    svc, store, audit, _ = setup()
    before = store.get('a')
    r = svc.ingest_patches([
        atlas(name=None),
        atlas(city=' '),
        atlas(),
        FeedEnvelope('Beacon', {'entityId':'a', 'timestamp':'2026-01-03T00:00:00Z','location':None}),
    ])
    assert (r.inserted,r.updated,r.stale,r.rejected) == (0,0,0,4)
    assert all(e.business_id is None for e in r.events)
    assert store.get('a') == before
    assert audit.count() == 4


def test_equal_timestamp_stale_and_per_record_order():
    svc, store, _, _ = setup()
    r = svc.ingest_patches([
        atlas(ts='2026-01-01T00:00:00Z', name='Ignored'),
        atlas(ts='2026-01-03T00:00:00Z', city='Paris'),
        atlas(ts='2026-01-02T00:00:00Z', name='Too Old'),
    ])
    assert [e.action for e in r.events] == ['stale','updated','stale']
    assert store.get('a').city == 'Paris' and store.get('a').name == 'Old Name'


def test_empty_batch_and_existing_level1_still_works():
    svc, store, _, counters = setup()
    r = svc.ingest_patches([])
    assert r.total == 0 and r.events == () and counters.batches == 1
    result = svc.ingest_batch([FeedEnvelope('atlas', {
        'business_id':'new','name':'Fresh','city':'LA','updated_at':'2026-01-04T00:00:00Z'
    })])
    assert result.inserted == 1 and store.get('new').name == 'Fresh'
    assert counters.batches == 2
