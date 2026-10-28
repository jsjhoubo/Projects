from __future__ import annotations
import json
from pathlib import Path
from .models import EvaluationCase

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_cases(tenant: str, name: str = "regression") -> list[EvaluationCase]:
    path = DATA_DIR / tenant / f"{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [EvaluationCase.model_validate(json.loads(line)) for line in handle if line.strip()]
