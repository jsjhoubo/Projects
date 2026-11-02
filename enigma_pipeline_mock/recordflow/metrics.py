"""Operational ingestion metrics and aggregate counters."""
from dataclasses import dataclass
from .domain import BatchResult

@dataclass
class IngestCounters:
    batches: int = 0
    inserted: int = 0
    updated: int = 0
    stale: int = 0
    rejected: int = 0

    def observe(self, result: BatchResult) -> None:
        self.batches += 1
        self.inserted += result.inserted
        self.updated += result.updated
        self.stale += result.stale
        self.rejected += result.rejected

    @property
    def processed(self) -> int:
        return self.inserted + self.updated + self.stale + self.rejected
