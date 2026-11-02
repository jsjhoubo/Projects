# LEVEL 2 — Partial Business Updates (Patch Merge)

## Context
Suppliers now send **partial corrections** after their full snapshots. A correction may change only `name` or only `city`. The other field must be preserved in the canonical store. Reuse the same `IngestionService`, `MemoryBusinessStore`, `MemoryAuditLog`, `IngestCounters`, `BatchResult`, and `AuditEvent` from Level 1.

## Implementation task
Add a public method to `IngestionService` in `recordflow/ingestion.py`:

```python
def ingest_patches(self, patches: Iterable[FeedEnvelope]) -> BatchResult:
    ...
```

`FeedEnvelope.supplier` must identify `Atlas` or `Beacon` (case/outer whitespace ignored).

Each `FeedEnvelope.payload` is a *partial patch* with the following fields:

| Supplier | Required ID | Required timestamp | Optional name | Optional city |
|---|---|---|---|---|
| Atlas | `business_id` | `updated_at` | `name` | `city` |
| Beacon | `entityId` | `timestamp` | `displayName` | `location.city` (see note) |

For Beacon, `location` is optional; if it is present, it must be a mapping containing `city`. Missing `location` means the city is not being changed. An explicit `location=None` is invalid.

**Required field normalization:** Reuse `normalize_id`, `normalize_name`, `normalize_city`, and `parse_timestamp` from `recordflow.validation`. No new normalization rules. The event `supplier` is the canonical lowercase `atlas` or `beacon` on valid patches. For a rejected patch use the input supplier if it is a string, otherwise the literal string `"<invalid>"`.

## Rules (process each input in order)
1. Patch requires an existing record in `self.store` with the normalized ID. If absent: `rejected`, no store change.
2. Patch timestamp must be **strictly greater** than the stored `updated_at`. Otherwise: `stale`, no store change.
3. A valid newer patch changes **only fields explicitly provided**. Missing fields keep stored values. An explicit `None` or blank string in a provided `name` or `city` is **invalid**; it does not mean “no change.”
4. A patch must provide at least one of the optional update fields (`name`/`city`). A timestamp-only patch is `rejected`.
5. On success: update the stored `BusinessRecord` with merged `name`, `city`, incoming UTC timestamp, and canonical supplier source; emit `updated` and increment `updated`.
6. For every input, emit exactly one `AuditEvent` in input order and append it to `self.audit`. Accepted or stale patches use the normalized business ID as `event.business_id`; rejected patches use `None`. For valid/stale patches event `supplier` is canonical; for rejected, follow the rule above. `detail` may be empty.
7. Return a `BatchResult` with accurate counts and an immutable tuple of events. For this method `inserted` is always `0`. Call `self.counters.observe(result)` **exactly once per call**, including empty input.
8. Validation failures and unknown suppliers reject only that patch, not the whole batch. Do not silently convert unexpected store/programming exceptions into `rejected`.
9. `ingest_batch` from Level 1 must continue working unchanged. No concurrency or rollback requirement.

## Full input/output example
Initial stored record:

```text
BusinessRecord(business_id='a', name='Old Name', city='NYC',
               updated_at=2026-01-01T00:00:00+00:00, source='atlas')
```

Input patches, in this exact order:

```python
[
  FeedEnvelope(' ATLAS ', {'business_id':' A ', 'updated_at':'2026-01-02T00:00:00Z', 'name':'New Name'}),
  FeedEnvelope('Atlas', {'business_id':'a', 'updated_at':'2026-01-01T12:00:00Z', 'city':'Boston'}),
  FeedEnvelope('Atlas', {'business_id':'missing', 'updated_at':'2026-01-03T00:00:00Z', 'name':'X'}),
]
```

Expected result: `BatchResult(inserted=0, updated=1, stale=1, rejected=1, events=(...))` with actions `['updated','stale','rejected']`. Final store for `a`: `name='New Name'`, `city='NYC'`, `updated_at=2026-01-02T00:00:00+00:00`, `source='atlas'`. No record created for `missing`. Three audit events appended; counters observe the single batch.

## Acceptance
Run `python -m pytest tests/test_level1.py tests/test_level2.py -q`. Do not modify provided test files. You may inspect all existing code and tests.

**This add-on includes no replacement for `recordflow/ingestion.py`; preserve your Level 1 implementation and add only the new method.**
