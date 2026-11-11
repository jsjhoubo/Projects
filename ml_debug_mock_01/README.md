# ML Evaluation Pipeline Debugging — Interview Mock 01

Standalone Python engineering exercise. Not an official Anthropic / OpenAI interview question.

**Timebox:** 45–60 minutes. CPU only. No credentials or paid APIs needed.

An ML evaluation pipeline sometimes reports suspiciously high accuracy or misinterprets valid answers. Investigate the call chain, repair the implementation and prove correctness using unit tests.

Run from this directory:
```bash
python -m pip install pytest
python -m pytest -q
```

Some tests are expected to fail before a fix. Preserve the public API; do not weaken test assertions.

Main modules: `ml_eval/models.py`, `ml_eval/parsing.py`, `ml_eval/scoring.py`, `ml_eval/pipeline.py`.
