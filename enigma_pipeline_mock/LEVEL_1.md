# LEVEL 1 — Supplier Feed Ingestion

## Business context
Our downstream business search system must maintain a canonical view of supplier-sourced businesses. A feed is submitted as a batch of events. Only valid business identities should reach the canonical store. Operations wants an exact result summary and an audit trail for every record.

The input may contain two supplier formats. `Atlas` uses fields `business_id`, `name`, `city`, `updated_at`. `Beacon` uses `entityId`, `displayName`, `location.city`, `timestamp`. Adapters already exist.

## Required behavior
Implement `IngestionService.ingest_batch(...)` in `recordflow/ingestion.py`. Preserve the public API.

For **each input record, in order**:
1. Parse the supplier payload using the existing adapter and validation layer.
2. Invalid payloads must not abort the batch. Record a `rejected` audit event and increment `rejected`.
3. When the canonical business ID is not yet present, insert the record; increment `inserted`, emit `inserted` audit event.
4. When that ID already exists, overwrite only when the incoming `updated_at` is strictly greater than the stored one; increment `updated`, emit `updated` audit event.
5. Otherwise keep the existing record; increment `stale`, emit `stale` audit event.
6. Return a `BatchResult` with accurate counts, preserving an `events` entry for each input record.

### Rules
- Supplier ID normalization and timestamp parsing are defined in `validation.py`.
- All accepted timestamps must be UTC-aware.
- Audit event IDs are business IDs for parsed records; rejected payloads have `business_id=None`.
- An invalid record is not partially committed.
- Equal timestamps are stale.
- This level is single-threaded; there is no transaction rollback requirement yet.

Initial Store:
  {}

Input:
  business_id = "A1"
  name = "ABC"
  updated_at = "2026-01-01T10:00:00Z"

Expected Output:
  inserted = 1
  updated = 0
  stale = 0
  rejected = 0
  events = [AuditEvent("inserted", "A1", "atlas")]

Expected Store After Ingestion:
  A1 -> BusinessRecord(...)


## Acceptance
Run `python -m pytest -q tests/test_level1.py`.

You may inspect all source files. Do not modify existing tests. There may be edge cases beyond visible tests.
