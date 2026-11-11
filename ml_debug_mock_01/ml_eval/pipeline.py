from .scoring import evaluate_batch

def run_evaluation(samples, generate, *, mode="multiple_choice"):
    if not samples:
        raise ValueError("empty sample list")
    outputs = list(generate(samples))
    return evaluate_batch(samples, outputs, mode)
