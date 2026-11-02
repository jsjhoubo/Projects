"""Append-only audit sink."""
from typing import Iterable, List
from .domain import AuditEvent

class MemoryAuditLog:
    def __init__(self):
        self._events: List[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self._events.append(event)

    def extend(self, events: Iterable[AuditEvent]) -> None:
        self._events.extend(events)

    def all_events(self) -> List[AuditEvent]:
        return list(self._events)

    def count(self) -> int:
        return len(self._events)
