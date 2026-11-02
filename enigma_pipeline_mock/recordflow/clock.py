"""Minimal clock utilities used by tests and integrations."""
from datetime import datetime, timezone

class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

class FixedClock:
    def __init__(self, timestamp: datetime):
        self.timestamp = timestamp

    def now(self) -> datetime:
        return self.timestamp
