"""SnapshotStore: local, deterministic research infrastructure primitive.

This component intentionally uses an in-process store; production migration
requires persistence and concurrency semantics to be explicitly defined.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping
from collections import defaultdict
import copy
import hashlib
import json
import threading
import uuid


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_digest(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Record:
    key: str
    tenant: str
    revision: int
    payload: dict
    created_at: str
    digest: str

    def as_dict(self) -> dict:
        return {"key": self.key, "tenant": self.tenant,
                "revision": self.revision,
                "payload": copy.deepcopy(self.payload),
                "created_at": self.created_at, "digest": self.digest}


class SnapshotStore:
    """Versioned snapshots records with tenant-scoped lookup and history.

    This is a low-level building block, *not* a full authorization boundary.
    It cannot guarantee durability across processes or deployments.
    """

    def __init__(self, *, max_records: int=100000):
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self._entries: dict[tuple[str, str], list[Record]] = defaultdict(list)
        self._max_records = max_records
        self._lock = threading.RLock()
        self._count = 0

    def put(self, tenant: str, key: str, payload: Mapping[str, Any],
            *, expected_revision: int | None=None) -> Record:
        self._validate_identity(tenant, key)
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        with self._lock:
            history = self._entries[(tenant, key)]
            current_revision = history[-1].revision if history else 0
            if expected_revision is not None and expected_revision != current_revision:
                raise ValueError("stale revision")
            if self._count >= self._max_records:
                raise OverflowError("capacity exhausted")
            frozen = copy.deepcopy(dict(payload))
            item = Record(key=key, tenant=tenant, revision=current_revision + 1,
                          payload=frozen, created_at=utcnow(), digest=stable_digest(frozen))
            history.append(item)
            self._count += 1
            return item

    @staticmethod
    def _validate_identity(tenant: str, key: str) -> None:
        if not tenant or not isinstance(tenant, str):
            raise ValueError("tenant must be nonempty string")
        if not key or not isinstance(key, str):
            raise ValueError("key must be nonempty string")

    def get(self, tenant: str, key: str, revision: int | None=None) -> Record | None:
        self._validate_identity(tenant, key)
        with self._lock:
            history = self._entries.get((tenant, key), [])
            if not history:
                return None
            if revision is None:
                return history[-1]
            return next((x for x in history if x.revision == revision), None)

    def history(self, tenant: str, key: str, limit: int | None=None) -> list[Record]:
        self._validate_identity(tenant, key)
        with self._lock:
            values = list(self._entries.get((tenant, key), []))
            if limit is not None:
                if limit < 0:
                    raise ValueError("limit cannot be negative")
                values = values[-limit:] if limit else []
            return values

    def list_keys(self, tenant: str, prefix: str="") -> list[str]:
        return sorted(key for (t, key) in self._entries if t == tenant and key.startswith(prefix))

    def count(self, tenant: str | None=None) -> int:
        with self._lock:
            if tenant is None:
                return self._count
            return sum(len(v) for (t, _), v in self._entries.items() if t == tenant)

    def compare(self, tenant: str, key: str, from_revision: int,
                to_revision: int) -> dict:
        left = self.get(tenant, key, from_revision)
        right = self.get(tenant, key, to_revision)
        if left is None or right is None:
            raise KeyError("revision not found")
        a, b = left.payload, right.payload
        return {"added": {k:b[k] for k in b.keys()-a.keys()},
                "removed": {k:a[k] for k in a.keys()-b.keys()},
                "changed": {k:{"before":a[k],"after":b[k]} for k in a.keys()&b.keys() if a[k]!=b[k]}}

    def export(self, tenant: str) -> list[dict]:
        with self._lock:
            return [item.as_dict() for (t, _), hist in self._entries.items()
                    if t == tenant for item in hist]

    def import_records(self, tenant: str, payloads: Iterable[Mapping[str, Any]],
                       *, key_field: str="key") -> int:
        total = 0
        for record in payloads:
            if key_field not in record:
                raise ValueError("missing key")
            key = str(record[key_field])
            self.put(tenant, key, record)
            total += 1
        return total

    def delete_tenant(self, tenant: str) -> int:
        with self._lock:
            keys = [key for key in self._entries if key[0] == tenant]
            removed = sum(len(self._entries[key]) for key in keys)
            for key in keys:
                del self._entries[key]
            self._count -= removed
            return removed

    def stats(self) -> dict[str, Any]:
        with self._lock:
            tenants = set(t for t, _ in self._entries)
            return {"records": self._count, "tenants": len(tenants),
                    "keys": len(self._entries), "capacity": self._max_records}
