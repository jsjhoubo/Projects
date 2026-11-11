from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Sample:
    index: int
    question: str
    target: str

@dataclass(frozen=True)
class ModelOutput:
    sample_index: int
    response: str

@dataclass(frozen=True)
class EvalResult:
    score: float
    n: int
    correct: int
    per_sample: tuple[dict, ...]
    mode: Literal["multiple_choice", "binary"]
