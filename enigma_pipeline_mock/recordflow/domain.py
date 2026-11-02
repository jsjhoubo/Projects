"""Immutable business identity and event representations."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Tuple

@dataclass(frozen=True)
class BusinessRecord:
    business_id: str
    name: str
    city: str
    updated_at: datetime
    source: str

@dataclass(frozen=True)
class FeedEnvelope:
    supplier: str
    payload: Mapping[str, Any]

@dataclass(frozen=True)
class AuditEvent:
    action: str
    business_id: Optional[str]
    supplier: str
    detail: str = ""

@dataclass(frozen=True)
class BatchResult:
    inserted: int
    updated: int
    stale: int
    rejected: int
    events: Tuple[AuditEvent, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.stale + self.rejected
