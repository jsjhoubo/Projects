"""Supplier-specific field extraction, returning canonical records."""
from typing import Any, Mapping
from .domain import BusinessRecord
from .errors import InvalidPayload, UnknownSupplier
from .validation import normalize_id, normalize_name, normalize_city, parse_timestamp

def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidPayload(f"{label}: expected object")
    return value

def parse_atlas(payload: Mapping[str, Any]) -> BusinessRecord:
    obj = _as_mapping(payload, "Atlas payload")
    return BusinessRecord(
        business_id=normalize_id(obj.get("business_id")),
        name=normalize_name(obj.get("name")),
        city=normalize_city(obj.get("city")),
        updated_at=parse_timestamp(obj.get("updated_at")),
        source="atlas",
    )

def parse_atlas_patch(payload: Mapping[str, Any]) -> BusinessRecord:
    obj = _as_mapping(payload, "Atlas payload")
    return BusinessRecord(
        business_id=normalize_id(obj.get("business_id")),
        name=normalize_name(obj.get("name")) if "name" in obj else None,
        city=normalize_city(obj.get("city")) if "city" in obj else None,
        updated_at=parse_timestamp(obj.get("updated_at")),
        source="atlas",
    )
    
def parse_beacon(payload: Mapping[str, Any]) -> BusinessRecord:
    obj = _as_mapping(payload, "Beacon payload")
    location = _as_mapping(obj.get("location"), "location")
    return BusinessRecord(
        business_id=normalize_id(obj.get("entityId")),
        name=normalize_name(obj.get("displayName")),
        city=normalize_city(location.get("city")),
        updated_at=parse_timestamp(obj.get("timestamp")),
        source="beacon",
    )

def parse_beacon_patch(payload: Mapping[str, Any]) -> BusinessRecord:
    obj = _as_mapping(payload, "Beacon payload")
    location = _as_mapping(obj.get("location"), "location")
    return BusinessRecord(
        business_id=normalize_id(obj.get("entityId")),
        name=normalize_name(obj.get("displayName")) if "displayName" in obj else None,
        city=normalize_city(location.get("city")) if "city" in location else None,
        updated_at=parse_timestamp(obj.get("timestamp")),
        source="beacon",
    )
    
def parse_supplier(supplier: str, payload: Mapping[str, Any]) -> BusinessRecord:
    if not isinstance(supplier, str):
        raise UnknownSupplier("supplier must be a string")
    mapping = {"atlas": parse_atlas, "beacon": parse_beacon}
    parser = mapping.get(supplier.strip().casefold())
    if parser is None:
        raise UnknownSupplier(f"unsupported supplier: {supplier}")
    return parser(payload)

def parse_supplier_patch(supplier: str, payload: Mapping[str, Any]) -> BusinessRecord:
    if not isinstance(supplier, str):
        raise UnknownSupplier("supplier must be a string")
    mapping = {"atlas": parse_atlas_patch, "beacon": parse_beacon_patch}
    parser = mapping.get(supplier.strip().casefold())
    if parser is None:
        raise UnknownSupplier(f"unsupported supplier: {supplier}")
    return parser(payload)
