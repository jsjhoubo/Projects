"""Application-level errors."""
class InvalidPayload(ValueError):
    """A supplier payload cannot be mapped to a canonical record."""

class UnknownSupplier(InvalidPayload):
    """The ingestion service does not support this supplier."""

class StoreConflict(RuntimeError):
    """A storage operation violated its contract."""
