# ML Evaluation Debugging — Extended Engineering Mock

This is an **independently constructed interview exercise**, not an official question from Anthropic, OpenAI, Resolution, or Distyl AI.

The project has a multi-module evaluation flow and algorithmic subsystem. The existing `ml_eval/parsing.py` and `ml_eval/scoring.py` are intentionally left in their initial state for your debugging task.

## Run locally

```bash
python -m pip install pytest
python -m pytest tests/test_regression.py -q
python -m pytest tests/ -q
```

No paid API calls or GPU required. Regression tests contain deterministic input corpora for classification, ranking, probability calibration, regression, statistical intervals, dataset splitting, temporal checks, consensus, group metrics and sequences.

## Interview goal

1. Map input/output and grading semantics.
2. Identify inconsistent scores in edge cases.
3. Correct defects without changing public API.
4. Confirm invariant tests and full regression suite.
5. Explain performance, behavior changes and test coverage.

**This repository intentionally includes a large and diverse regression corpus. Test-line count is not equivalent to novel implementation complexity.** All modules are small enough to inspect individually; your challenge is navigating interactions across them.

Do not change test assertions merely to force a pass.

The task can be completed in multiple iterations; 45–60 minutes is the initial interview timebox.
