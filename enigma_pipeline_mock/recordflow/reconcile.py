"""Non-mutating reconciliation helpers for downstream quality monitors."""
from collections import Counter
from typing import Iterable, Mapping
from .domain import BusinessRecord

def city_histogram(rows: Iterable[BusinessRecord]) -> Mapping[str, int]:
    return dict(Counter(row.city.casefold() for row in rows))

def source_histogram(rows: Iterable[BusinessRecord]) -> Mapping[str, int]:
    return dict(Counter(row.source for row in rows))

def stale_ids(current: Iterable[BusinessRecord], proposed: Iterable[BusinessRecord]):
    existing = {row.business_id: row.updated_at for row in current}
    return [row.business_id for row in proposed
            if row.business_id in existing and row.updated_at <= existing[row.business_id]]
