from .models import EvalResult
from .parsing import extract_answer

def evaluate_batch(samples, outputs, mode="multiple_choice"):
    if mode not in ("multiple_choice", "binary"):
        raise ValueError("unknown mode")
    sample_map = {sample.index: sample for sample in samples}
    if len(sample_map) != len(samples):
        raise ValueError("duplicate sample index")
    details = []
    correct = 0
    for out in outputs:
        sample = sample_map[out.sample_index]
        target = sample.target.upper() if mode == "multiple_choice" else sample.target.lower()
        pred = extract_answer(out.response, mode)
        passed = pred == target
        correct += int(passed)
        details.append({"index": out.sample_index, "pred": pred, "target": target, "correct": passed})
    n = len(outputs)
    return EvalResult(
        score=correct/n if n else 0.0,
        n=n,
        correct=correct,
        per_sample=tuple(details),
        mode=mode,
    )
