"""Generate deterministic point-in-time ML features from supplier snapshots.

LEVEL 4: This implementation intentionally contains silent correctness bugs.
Fix this file without changing its public API or the supplied tests.
"""
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .domain import BusinessRecord


@dataclass(frozen=True)
class TrainingExample:
    business_id: str
    label_time: datetime
    city: str | None
    source: str | None
    record_age_hours: float | None


def build_training_examples(
    snapshots: Iterable[BusinessRecord],
    labels: Iterable[tuple[str, datetime]],
) -> list[TrainingExample]:
    """See LEVEL_4.md for the exact as-of join contract."""
    id2snapshots = {}
    for r in snapshots:
        if r.business_id not in id2snapshots:
            id2snapshots[r.business_id] = []
        id2snapshots[r.business_id].append(r)

    for id in id2snapshots.keys():
        id2snapshots[id] =sorted(id2snapshots[id], key= lambda x: x.updated_at)
    result = []

    for business_id, label_time in labels:
        list_snapshots = id2snapshots.get(business_id)
        
        if list_snapshots is None:
            result.append(TrainingExample(business_id, label_time, None, None, None))
            continue
        
        
        
        index = bisect_right(list_snapshots, label_time, key =lambda x: x.updated_at)
        if index -1 <0 :
            result.append(TrainingExample(business_id, label_time, None, None, None))
            continue
        r =list_snapshots[index-1]            
        result.append(
            TrainingExample(
                business_id=business_id,
                label_time=label_time,
                city=r.city,
                source=r.source,
                record_age_hours=(label_time - r.updated_at).total_seconds() / 3600,
            )
        )
    return result
