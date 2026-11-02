"""Canonical record store boundary.

MemoryBusinessStore is intentionally small, but its API mirrors one that could
be replaced by a persistent database implementation.
"""
from typing import Dict, Iterable, List, Optional
from .domain import BusinessRecord
from .errors import StoreConflict

class MemoryBusinessStore:
    def _updateIndex(self, record:BusinessRecord):
        city = record.city.strip().casefold()
        if city not in self._city_index:
            self._city_index[city] =set()
        self._city_index[city].add(record.business_id)
        source = record.source.strip().casefold()
        if source not in self._source_index:
            self._source_index[source] =set()
        self._source_index[source].add(record.business_id)

    def _remove_from_indexes(self, record: BusinessRecord) -> None:
        for index, value in (
            (self._city_index, record.city),
            (self._source_index, record.source),
        ):
            key = value.strip().casefold()
            ids = index.get(key)
            if ids is not None:
                ids.discard(record.business_id)
                if not ids:
                    del index[key]
        
    def __init__(self, seed: Iterable[BusinessRecord] = ()):
        self._rows: Dict[str, BusinessRecord] = {}
        self._city_index: Dict[str, set] ={}
        self._source_index: Dict[str, set] ={}
        for record in seed:
            self.put(record)

    def get(self, business_id: str) -> Optional[BusinessRecord]:
        return self._rows.get(business_id)

    def put(self, record: BusinessRecord) -> None:
        if not record.business_id:
            raise StoreConflict("empty business ID")
        old_record = self._rows.get(record.business_id)
        if old_record is not None:
            self._remove_from_indexes(old_record)
        self._rows[record.business_id] = record
        self._updateIndex(record)
        
    def count(self) -> int:
        return len(self._rows)

    def all_records(self) -> List[BusinessRecord]:
        return list(self._rows.values())

    def snapshot(self) -> Dict[str, BusinessRecord]:
        return dict(self._rows)

    def restore(self, snapshot: Dict[str, BusinessRecord]) -> None:
        self._rows = dict(snapshot)
        self._city_index ={}
        self._source_index ={}
        for id, record in self._rows.items():
            self._updateIndex(record)
