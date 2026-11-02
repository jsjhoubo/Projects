"""Supplier feed ingestion coordination.

IMPLEMENTATION TASK: ingest_batch is incomplete. Read LEVEL_1.md.
"""

from typing import Iterable

from recordflow.suppliers import parse_supplier, parse_supplier_patch
from .domain import AuditEvent, BusinessRecord, FeedEnvelope, BatchResult
from .errors import InvalidPayload
from .storage import MemoryBusinessStore
from .audit import MemoryAuditLog
from .metrics import IngestCounters


class IngestionService:
    def __init__(
        self,
        store: MemoryBusinessStore,
        audit: MemoryAuditLog,
        counters: IngestCounters | None = None,
    ):
        self.store = store
        self.audit = audit
        self.counters = counters if counters is not None else IngestCounters()

    def ingest_batch(self, records: Iterable[FeedEnvelope]) -> BatchResult:
        """Apply one supplier batch in input order. See LEVEL_1.md."""

        inserted = 0
        updated = 0
        rejected = 0
        stale = 0
        events = ()
        
        for record in records:
            supplier = record.supplier
            business_id = None

            try:
                business_record = parse_supplier(record.supplier, record.payload)
                business_id = business_record.business_id
                if business_id is not None:

                    old_business_record = self.store.get(business_record.business_id)
                    if old_business_record is None:
                        action = "inserted"
                        inserted += 1
                        self.store.put(business_record)
                    else:
                        if (
                            business_record.updated_at is not None
                            and business_record.updated_at
                            > old_business_record.updated_at
                        ):
                            action = "updated"
                            updated += 1
                            self.store.put(business_record)
                        else:
                            action = "stale"
                            stale += 1
                else:
                    action = "rejected"
                    rejected += 1

            except Exception as e:
                
                rejected += 1
                action = "rejected"
            event = AuditEvent(
                action=action, business_id=business_id, supplier=supplier
            )
            self.audit.append(event)
            events = events + (event,)

        ret = BatchResult(
            inserted=inserted,
            updated=updated,
            stale=stale,
            rejected=rejected,
            events=events,
        )
        self.counters.observe(ret)
        return ret

    def _parse_patch(self, patch: FeedEnvelope) -> BusinessRecord:
        record = parse_supplier_patch(patch.supplier, patch.payload)
        if record.name is None and record.city is None:
            raise InvalidPayload("patch requires name or city")
        return record

    def _rejected_patch(self, patch: FeedEnvelope) -> AuditEvent:
        supplier = patch.supplier if isinstance(patch.supplier, str) else "<invalid>"
        return AuditEvent(action="rejected", business_id=None, supplier=supplier)

    def _merge_patch(self, old: BusinessRecord, patch: BusinessRecord) -> None:
        self.store.put(BusinessRecord(
            business_id=old.business_id,
            name=old.name if patch.name is None else patch.name,
            city=old.city if patch.city is None else patch.city,
            updated_at=patch.updated_at,
            source=patch.source,
        ))

    def _apply_patch(self, patch: FeedEnvelope) -> AuditEvent:
        try:
            record = self._parse_patch(patch)
        except InvalidPayload:
            return self._rejected_patch(patch)

        old = self.store.get(record.business_id)
        if old is None:
            return self._rejected_patch(patch)
        if record.updated_at <= old.updated_at:
            return AuditEvent("stale", record.business_id, record.source)
        self._merge_patch(old, record)
        return AuditEvent("updated", record.business_id, record.source)

    def ingest_patches(self, patches: Iterable[FeedEnvelope]) -> BatchResult:
        counts = {"updated": 0, "stale": 0, "rejected": 0}
        events = []
        for patch in patches:
            event = self._apply_patch(patch)
            self.audit.append(event)
            events.append(event)
            counts[event.action] += 1
        result = BatchResult(inserted=0, events=tuple(events), **counts)
        self.counters.observe(result)
        return result
