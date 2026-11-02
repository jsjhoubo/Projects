from datetime import datetime, timezone
from recordflow.ingestion import IngestionService
from recordflow.storage import MemoryBusinessStore
from recordflow.audit import MemoryAuditLog
from recordflow.metrics import IngestCounters
from recordflow.domain import FeedEnvelope


def atlas(id, name="ABC Market", city="NYC", time="2026-10-25T12:00:00Z"):
    return FeedEnvelope("Atlas", {"business_id": id, "name": name, "city": city, "updated_at": time})


def beacon(id, name="Beacon Cafe", city="Boston", time=1792933200):
    return FeedEnvelope("BEACON", {"entityId": id, "displayName": name,
                                   "location": {"city": city}, "timestamp": time})


def setup():
    store, audit, counters = MemoryBusinessStore(), MemoryAuditLog(), IngestCounters()
    return IngestionService(store, audit, counters), store, audit, counters


def test_insert_two_formats():
    svc, store, audit, counters = setup()
    result = svc.ingest_batch([atlas(" Co-1 "), beacon("Co-2")])
    assert (result.inserted, result.updated, result.stale, result.rejected) == (2, 0, 0, 0)
    assert [e.action for e in result.events] == ["inserted", "inserted"]
    assert result.total == 2 and store.count() == 2
    assert store.get("co-1").name == "ABC Market"
    assert store.get("co-2").source == "beacon"
    assert store.get("co-2").updated_at.tzinfo is not None
    assert audit.count() == 2 and counters.processed == 2 and counters.batches == 1


def test_newer_update_and_equal_stale():
    svc, store, audit, _ = setup()
    result = svc.ingest_batch([
        atlas("x", name="First", time="2026-01-01T00:00:00Z"),
        atlas("X", name="New", time="2026-01-02T00:00:00Z"),
        atlas("x", name="Ignored", time="2026-01-02T00:00:00+00:00"),
        atlas("x", name="Old", time="2025-01-01T00:00:00Z"),
    ])
    assert (result.inserted, result.updated, result.stale, result.rejected) == (1, 1, 2, 0)
    assert [e.action for e in result.events] == ["inserted", "updated", "stale", "stale"]
    assert store.get("x").name == "New" and audit.count() == 4


def test_bad_record_does_not_abort_batch():
    svc, store, audit, _ = setup()
    result = svc.ingest_batch([
        atlas("ok"),
        atlas(" "),
        FeedEnvelope("Beacon", {"entityId": "b", "displayName": "B", "location": None, "timestamp": 1}),
        atlas("later"),
    ])
    assert (result.inserted, result.updated, result.stale, result.rejected) == (2, 0, 0, 2)
    assert [e.action for e in result.events] == ["inserted", "rejected", "rejected", "inserted"]
    assert audit.count() == 4 and all(e.business_id is None for e in result.events if e.action == "rejected")
    assert set(store.snapshot()) == {"ok", "later"}


def test_empty_and_multiple_batches():
    svc, store, audit, counters = setup()
    empty = svc.ingest_batch([])
    another = svc.ingest_batch([atlas("z")])
    assert empty.total == 0 and empty.events == ()
    assert counters.batches == 2 and counters.inserted == 1
    assert another.total == 1 and audit.count() == 1


def test_reject_unknown_supplier_and_naive_datetime():
    svc, store, audit, _ = setup()
    result = svc.ingest_batch([
        FeedEnvelope("unknown", {"business_id": "abc"}),
        atlas("n", time="2026-01-01T10:00:00"),
        atlas("good", time="2026-01-01T10:00:00+01:00"),
    ])
    assert result.rejected == 2 and result.inserted == 1
    assert store.get("good").updated_at == datetime(2026, 1, 1, 9, tzinfo=timezone.utc)
    assert [e.business_id for e in result.events] == [None, None, "good"]
