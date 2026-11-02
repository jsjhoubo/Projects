"""Level 3: consistent cached read queries with store updates."""
from datetime import datetime, timezone
from recordflow.domain import BusinessRecord
from recordflow.storage import MemoryBusinessStore
from recordflow.query import BusinessQuery

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 2, tzinfo=timezone.utc)


def rec(id, city, source, t=T0):
    return BusinessRecord(id, id.upper(), city, t, source)


def test_basic_query_order_and_fresh_list():
    a, b, c = rec('a', 'Boston', 'atlas'), rec('b', 'Boston', 'beacon'), rec('c', 'NYC', 'atlas')
    store = MemoryBusinessStore([a, b, c])
    query = BusinessQuery(store)
    first = query.by_city(' bOsToN ')
    assert first == [a, b]
    first.clear()
    assert query.by_city('Boston') == [a, b]
    assert query.by_supplier(' ATLAS ') == [a, c]
    assert query.by_city('unknown') == []
    assert query.by_id(' A ') == a


def test_put_overwrite_and_insertion_are_visible():
    a, b, c = rec('a', 'Boston', 'atlas'), rec('b', 'Boston', 'beacon'), rec('c', 'NYC', 'atlas')
    store = MemoryBusinessStore([a, b, c]); query = BusinessQuery(store)
    assert query.by_city('Boston') == [a, b]
    new_a = rec('a', 'NYC', 'beacon', T1)
    store.put(new_a)
    assert query.by_city('Boston') == [b]
    assert query.by_city('nyc') == [new_a, c]
    assert query.by_supplier('atlas') == [c]
    assert query.by_supplier('beacon') == [new_a, b]
    d = rec('d', 'Boston', 'atlas', T1)
    store.put(d)
    assert query.by_city('Boston') == [b, d]


def test_restore_replaces_cached_view():
    a, b = rec('a', 'Boston', 'atlas'), rec('b', 'NYC', 'beacon')
    store = MemoryBusinessStore([a, b]); query = BusinessQuery(store)
    assert query.by_city('Boston') == [a]
    store.restore({'b': b, 'c': rec('c', 'Boston', 'atlas')})
    assert query.by_city('Boston') == [rec('c', 'Boston', 'atlas')]
    assert query.by_supplier('beacon') == [b]


def test_no_second_full_store_scan_without_mutation():
    class CountedStore(MemoryBusinessStore):
        def __init__(self, seed):
            super().__init__(seed)
            self.scans = 0
        def all_records(self):
            self.scans += 1
            return super().all_records()

    store = CountedStore([rec(str(i), 'Boston' if i % 2 else 'NYC', 'atlas' if i % 3 else 'beacon') for i in range(100)])
    query = BusinessQuery(store)
    assert len(query.by_city('Boston')) == 50
    scans_after_first = store.scans
    assert scans_after_first <= 1
    for _ in range(5):
        assert len(query.by_city('nyc')) == 50
        assert len(query.by_supplier('atlas')) == 66
    assert store.scans == scans_after_first
    store.put(rec('new', 'Boston', 'atlas'))
    assert len(query.by_city('Boston')) == 51
    scans_after_mutation = store.scans
    for _ in range(5):
        query.by_supplier('atlas')
    assert store.scans == scans_after_mutation
