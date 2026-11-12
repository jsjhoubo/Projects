"""Training/evaluation splitting for event-level and entity-level datasets.

The APIs deliberately expose separate policies rather than hiding them in one
generic split(). Review against the challenge spec before changing behavior.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from datetime import timedelta
import random


def group_stratified_split(records, *, group_key="user_id", label_key="label",
                           train_fraction=0.8, seed=17):
    """Random GROUP-level assignment (baseline, NOT guaranteed stratification).

    Discussion / contract (2026-11-11):
    - Each record has exactly one discrete label; a group has a label histogram.
    - Hard constraint: one group goes wholly to train OR wholly to validation.
      No records may be duplicated or dropped, and partition order is stable.
    - Desired train sample ratio and per-label proportions are SOFT objectives.
      With skewed group sizes or group label histograms, even the globally
      optimal binary assignment may not achieve them simultaneously.
    - This baseline chooses approximately train_fraction of GROUP IDs, not
      train_fraction of RECORDS. That is a valid group-preserving split, but
      its name must not be interpreted as a stratification guarantee.
    - If stronger stratification is required, define an explicit optimization
      objective and acceptable tolerances before changing the implementation.
      Never silently weaken group integrity or resample validation rows.

    The current baseline implementation is deliberately unchanged so existing
    debugging work can continue. Returns train, validation in input order.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    records = list(records)
    groups = defaultdict(list)
    for record in records:
        groups[record[group_key]].append(record)
    rng = random.Random(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    desired = int(len(records) * train_fraction)
    train_groups = set(group_ids[:int(len(group_ids) * train_fraction)])
    train = [r for r in records if r[group_key] in train_groups]
    val = [r for r in records if r[group_key] not in train_groups]
    return train, val


def temporal_event_split(records, cutoff, *, time_key="event_time"):
    """Events before cutoff go to train; cutoff and later go to validation."""
    train, validation = [], []
    for record in records:
        if record[time_key] <= cutoff:
            train.append(record)
        else:
            validation.append(record)
    return train, validation


def purged_temporal_split(records, cutoff, *, time_key="event_time",
                          label_end_key="label_available_at", embargo=timedelta(0)):
    """Prevent training examples whose labels are not available by cutoff.

    Train events must occur before the cutoff and have labels fully available
    no later than cutoff. Validation events start at cutoff + embargo.
    Events between those boundaries are excluded.
    """
    if embargo < timedelta(0):
        raise ValueError("embargo cannot be negative")
    train, val = [], []
    for record in records:
        t = record[time_key]
        if t < cutoff:
            train.append(record)
        elif t >= cutoff + embargo:
            val.append(record)
    return train, val


def build_group_folds(records, n_folds=5, *, group_key="user_id", seed=17):
    """Map each record to a fold, keeping any one group within one fold.

    The largest and smallest folds should differ by at most the largest
    group size, when evaluated on record counts.
    """
    if n_folds < 2:
        raise ValueError("at least two folds required")
    records = list(records)
    unique = list(dict.fromkeys(record[group_key] for record in records))
    random.Random(seed).shuffle(unique)
    assigned = {group: i % n_folds for i, group in enumerate(unique)}
    return [assigned[record[group_key]] for record in records]


def rolling_origin_splits(records, cutoffs, *, time_key="event_time",
                          horizon=timedelta(days=7)):
    """Produce one chronological train/validation split per cutoff.

    Validation is [cutoff, cutoff + horizon). No event may appear in both.
    Every split is deterministic and independent of input order.
    """
    if horizon <= timedelta(0):
        raise ValueError("horizon must be positive")
    records = list(records)
    for cutoff in cutoffs:
        train = [r for r in records if r[time_key] < cutoff]
        validation = [r for r in records if r[time_key] <= cutoff + horizon]
        yield train, validation
