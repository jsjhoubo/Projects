"""Read-only queries over the canonical business view."""
from typing import List, Optional, Mapping, Set
from .domain import BusinessRecord
from .storage import MemoryBusinessStore

class BusinessQuery:
    def __init__(self, store: MemoryBusinessStore):
        self._store = store

    def by_id(self, business_id: str) -> Optional[BusinessRecord]:
        return self._store.get(business_id.strip().casefold())

    def by_city(self, city: str) -> List[BusinessRecord]:
        return self._by_index(city, self._store._city_index, "city")
    
    def by_supplier(self, supplier: str) -> List[BusinessRecord]:
        return self._by_index(supplier, self._store._source_index, "source")

    def _by_index(
        self, value: str, index: Mapping[str, Set[str]], field: str
    ) -> List[BusinessRecord]:
        needle = value.strip().casefold()
        ret = []
        for business_id in sorted(index.get(needle, ())):
            record = self._store.get(business_id)
            if record is not None and getattr(record, field).strip().casefold() == needle:
                ret.append(record)
        return ret
    
