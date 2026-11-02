# LEVEL 3 — Fast Business Lookup (Performance & Correctness)

## Business context
The RecordFlow canonical store now contains hundreds of thousands of businesses. Read-side analytics frequently filter by city or by supplier. The current `BusinessQuery.by_city()` and `by_supplier()` call `store.all_records()` and scan every record per query, which is slow when queries repeat.

Your task is to optimize these **two existing query methods** while keeping their public signatures and returned values compatible with Level 1 and Level 2. **Do not implement a new ingestion method.**

## Existing interfaces
Read these files first:
- `recordflow/storage.py`: `MemoryBusinessStore` (`get`, `put`, `all_records`, `snapshot`, `restore`).
- `recordflow/query.py`: `BusinessQuery` (`by_id`, `by_city`, `by_supplier`).
- `recordflow/domain.py`: immutable `BusinessRecord`.

The store remains the authoritative source of current records. `IngestionService.ingest_batch()` and `ingest_patches()` already call `store.put()`. You may edit `storage.py` and `query.py`, but **do not change the public signatures** of any existing method.

## Required input and output contracts
- `BusinessQuery.by_city(city: str) -> list[BusinessRecord]`: return all *current* records whose `record.city.casefold()` equals `city.strip().casefold()`.
- `BusinessQuery.by_supplier(supplier: str) -> list[BusinessRecord]`: return all *current* records whose `record.source` equals `supplier.strip().casefold()`.
- Preserve the output order of `store.all_records()` (which is Python dictionary insertion order by business ID). Replacing a row via `put()` does not change its insertion position; `restore()` preserves its snapshot's key order.
- Empty result is `[]`. Return a fresh list: modifying the returned list must not affect future queries.
- `by_id()` behavior stays unchanged.
- You may assume query inputs are strings; no concurrency, persistence, rollback, or deletes are required.

## Required performance behavior
Repeated queries with **no store changes** must not repeatedly scan all records. A solution may build a lazy index or rebuild a cached index when the store changes.

Specifically:
1. The first indexed query against a store may scan the entire store to build lookup information.
2. Subsequent `by_city()`/`by_supplier()` calls with **no intervening mutation** must not call `store.all_records()` again.
3. A `store.put()` (insert or overwrite) or `store.restore()` must be reflected in the **next** query on an already-created `BusinessQuery` object. Both old and new city/supplier values must be correct; stale records must not be returned.
4. Querying by city and then by supplier without mutation should reuse the same cached store traversal, not perform two full scans.
5. Aim for `O(N)` index construction/rebuild after a mutation and `O(M)` per stable query returning `M` records (excluding normalization). **No requirement** to support `O(1)` incremental updates or avoid a rebuild after every mutation.

## Example
Initial store insertion order:

```
BusinessRecord('a', 'A Cafe', 'Boston', t0, 'atlas')
BusinessRecord('b', 'B Shop', 'Boston', t0, 'beacon')
BusinessRecord('c', 'C Deli', 'NYC', t0, 'atlas')
```

Expected:

```
query.by_city(' bOsToN ')       -> [record_a, record_b]
query.by_supplier('ATLAS')      -> [record_a, record_c]
```

Then do `store.put(BusinessRecord('a', 'A Cafe', 'NYC', t1, 'beacon'))` (same ID, updated fields). Expected:

```
query.by_city('Boston')        -> [record_b]
query.by_city('nyc')           -> [new_record_a, record_c]
query.by_supplier('atlas')     -> [record_c]
query.by_supplier('beacon')    -> [new_record_a, record_b]
```

The order is **original store insertion order**, not alphabetical order and not update time order.

## Acceptance
- Keep Level 1 and Level 2 behavior unchanged.
- Do not change the supplied tests. Run:

```powershell
python -m pytest tests/test_existing_components.py tests/test_level1.py tests/test_level2.py tests/test_level3.py -v
```

- If your local Level 2 is unfinished, you may run only `tests/test_level3.py` to evaluate this task independently.
- The included `tests/test_level3.py` checks functional behavior and ensures repeated stable queries reuse prior work. This level deliberately does **not** add ML model behavior or hidden business rules.

Only `LEVEL_3.md` and `tests/test_level3.py` are included in this add-on. Keep your existing Python implementation files; modify them locally.
