"""Deterministic wash trading signal extraction.

Signals are *not* policy decisions. Callers must apply tenant-specific rules,
thresholds, and effective-date gates. Keep matches auditable for review.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence
import re

CATEGORY = 'wash_trading'
DOMAIN = 'financial'
PATTERNS = ['wash\\s+trad\\w*']

@dataclass(frozen=True)
class Signal:
    category: str
    domain: str
    matched_text: str
    start: int
    end: int
    confidence: float
    source: str = "regex"

    def serialize(self) -> dict:
        return asdict(self)


class Detector:
    def __init__(self, patterns: Sequence[str] | None = None, *, case_sensitive=False,
                 max_matches=32, minimum_length=2):
        if max_matches < 1:
            raise ValueError("max_matches must be positive")
        flags = 0 if case_sensitive else re.IGNORECASE
        self.expressions = tuple(re.compile(p, flags) for p in (patterns or PATTERNS))
        self.max_matches = max_matches
        self.minimum_length = minimum_length

    def inspect(self, text: str) -> list[Signal]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        found = []
        for expression in self.expressions:
            for match in expression.finditer(text):
                if len(match.group()) < self.minimum_length:
                    continue
                found.append(Signal(
                    category=CATEGORY, domain=DOMAIN,
                    matched_text=match.group(), start=match.start(), end=match.end(),
                    confidence=self._confidence(match.group(), expression.pattern)))
                if len(found) >= self.max_matches:
                    return self._deduplicate(found)
        return self._deduplicate(found)

    @staticmethod
    def _deduplicate(signals: Iterable[Signal]) -> list[Signal]:
        unique = {}
        for item in signals:
            key = (item.start, item.end, item.category)
            if key not in unique or unique[key].confidence < item.confidence:
                unique[key] = item
        return sorted(unique.values(), key=lambda s: (s.start, -s.confidence))

    @staticmethod
    def _confidence(value: str, regex: str) -> float:
        # Stable heuristic score for the deterministic test fixture.
        length_bonus = min(len(value) / 50.0, 0.2)
        specificity = min(len(regex) / 200.0, 0.15)
        return round(min(0.98, 0.60 + length_bonus + specificity), 4)

    def summarize(self, text: str) -> Mapping[str, object]:
        signals = self.inspect(text)
        return {"category": CATEGORY, "domain": DOMAIN,
                "count": len(signals),
                "max_confidence": max((x.confidence for x in signals), default=0),
                "spans": [{"start": x.start, "end": x.end} for x in signals]}

    def redact(self, text: str, replacement="[REDACTED]") -> str:
        spans = self.inspect(text)
        if not spans:
            return text
        out = []
        cursor = 0
        for item in spans:
            if item.start < cursor:
                continue
            out.extend((text[cursor:item.start], replacement))
            cursor = item.end
        out.append(text[cursor:])
        return "".join(out)

    def batches(self, texts: Iterable[str]) -> list[Mapping[str, object]]:
        return [self.summarize(text) for text in texts]


def check(text: str, *, threshold: float=0.5) -> bool:
    return any(signal.confidence >= threshold for signal in Detector().inspect(text))


def get_default_detector() -> Detector:
    return Detector()
