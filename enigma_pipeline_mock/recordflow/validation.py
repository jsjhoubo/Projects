"""Canonical field normalization; not ingestion policy."""
from datetime import datetime, timezone
from typing import Any
from .errors import InvalidPayload

def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidPayload(f"{field}: expected string")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise InvalidPayload(f"{field}: must be nonempty")
    return normalized

def normalize_id(value: Any) -> str:
    result = require_text(value, "business_id")
    if len(result) > 64:
        raise InvalidPayload("business_id: too long")
    return result.casefold()

def normalize_name(value: Any) -> str:
    result = require_text(value, "name")
    if len(result) > 200:
        raise InvalidPayload("name: too long")
    return result

def normalize_city(value: Any) -> str:
    result = require_text(value, "city")
    if len(result) > 120:
        raise InvalidPayload("city: too long")
    return result

def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, bool):
        raise InvalidPayload("updated_at: invalid boolean")
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise InvalidPayload("updated_at: invalid epoch") from exc
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InvalidPayload("updated_at: invalid ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise InvalidPayload("updated_at: timezone required")
        return parsed.astimezone(timezone.utc)
    raise InvalidPayload("updated_at: unsupported type")
