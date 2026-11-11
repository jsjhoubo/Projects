"""Dependency-free numerical evaluation primitives.

Public functions accept Python sequences and return deterministic results.
Input validation is explicit; nothing depends on a hosted model or GPU.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

def _pairs(values, labels):
    a,b=list(values),list(labels)
    if len(a)!=len(b): raise ValueError("length mismatch")
    return a,b

def _safe_div(a,b,zero=0.0):
    return a/b if b else float(zero)

def _finite(values):
    out=[float(x) for x in values]
    if any(not math.isfinite(x) for x in out): raise ValueError("non-finite value")
    return out

def _check_binary(labels):
    out=[int(x) for x in labels]
    if any(x not in (0,1) for x in out): raise ValueError("binary labels required")
    return out

# Include internal helpers when analytics modules use `from ._core import *`.
__all__ = ["math", "Counter", "defaultdict", "mean", "_pairs", "_safe_div", "_finite", "_check_binary"]
