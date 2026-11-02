from recordflow.validation import normalize_id, parse_timestamp
from recordflow.domain import BusinessRecord
from recordflow.storage import MemoryBusinessStore
from recordflow.query import BusinessQuery
from datetime import datetime, timezone

def test_normalization():
    assert normalize_id("  A-B  ") == "a-b"
    assert parse_timestamp("2026-01-01T00:00:00Z") == datetime(2026, 1, 1, tzinfo=timezone.utc)

def test_query_store():
    r = BusinessRecord("a", "A", "New York", datetime.now(timezone.utc), "atlas")
    store = MemoryBusinessStore([r])
    q = BusinessQuery(store)
    assert q.by_id(" A ") == r
    assert q.by_city("NEW YORK") == [r]
