"""Offline feed JSON utilities used by support tooling."""
import json
from pathlib import Path
from typing import Iterable, List
from .domain import FeedEnvelope
from .errors import InvalidPayload

def load_jsonl(path: str | Path) -> List[FeedEnvelope]:
    result = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidPayload(f"line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise InvalidPayload(f"line {line_number}: expected object")
            result.append(FeedEnvelope(supplier=value.get("supplier"), payload=value.get("payload")))
    return result

def dump_summary(result, path: str | Path):
    output = {"inserted": result.inserted, "updated": result.updated,
              "stale": result.stale, "rejected": result.rejected,
              "events": [{"action": e.action, "business_id": e.business_id,
                          "supplier": e.supplier, "detail": e.detail}
                         for e in result.events]}
    Path(path).write_text(json.dumps(output, indent=2), encoding="utf-8")
