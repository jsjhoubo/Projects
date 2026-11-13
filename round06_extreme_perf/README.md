# Round 06 — Extreme ML Systems Performance Mock (8k–12k LOC)

Original interview practice code. **Not an official, recovered, or leaked task**
from Resolution, OpenAI, or Anthropic. The design combines the verified
Resolution interview format (read a low-quality PyTorch Colab and explain
performance improvements) with harder implementation, differential regression,
profiling and benchmark requirements.

## Google Colab GPU workbench

Open [01_gpu_workbench.ipynb](notebooks/01_gpu_workbench.ipynb) in Google Colab
([launch Colab](https://colab.research.google.com/github/jsjhoubo/Projects/blob/interview/06-extreme-perf/round06_extreme_perf/notebooks/01_gpu_workbench.ipynb)).
Select **Runtime → Change runtime type → GPU** first, then run cells to
clone the full repo, check CUDA, run regression tests, profile attention/training,
and save measurements. This notebook launches the full codebase instead of
flattening 8,404 Python LOC into one notebook.
For the blind review round use [00_blind_review.ipynb](notebooks/00_blind_review.ipynb).

## What you are handed

You inherit a small causal language-model training and serving repository.
Most units produce mathematically legitimate outputs but make expensive runtime
choices. The system has multi-module data flow: corpus -> sampler -> batching
-> causal decoder -> optimizer -> validation -> generation. Low-level tensor
microbenchmarks and 42 distinct scenario workbooks supplement the end-to-end
model. Tests are correctness gates rather than reference solutions.

The repo is intentionally much larger than a 45-minute interview. The
interviewer will point you to a subset, but you have to navigate the rest.

## The integrated exam

### Part 1 — 45 minutes, code reading only (Resolution format)

- Read `notebooks/00_blind_review.ipynb` without AI, execution or edits.
- Trace real tensor shapes and memory layout from input tokens to next-token
  loss and backward computation.
- Identify and rank five performance issues on CPU and GPU separately.
- Give a concrete hypothesis, measurement, proposed fix, expected impact,
  regression risk and validation check for each.
- Discuss GPU synchronization, kernel launch, host-to-device transfers,
  dense causal attention and hidden quadratic allocations.
- State which transformations preserve numerical behavior and which only
  approximate it.

### Part 2 — Hands-on rescue (OpenAI-inspired)

- Optimize the code you inherited without rewriting the public API.
- Train on variable-length records and compare optimizer updates, not only
  logits. Preserve ignore-index semantics and causal information flow.
- Implement and validate an alternate attention execution path as a control.
- Investigate CUDA memory usage, activation checkpointing, mixed precision,
  device transfer, gradient accumulation and optimizer overhead.
- Measure sustained throughput, warmup effects, and p50/p95 step latency.
- Include the failure modes of compiled graphs, dropout and dynamic shape.

### Part 3 — Extreme performance engineering (Anthropic-inspired rigor)

- Select five of the 42 `scenarios/` kernels whose bottleneck changes with
  batch size, sequence width or distribution skew. Optimize them on equal
  hardware and identical test data. Do not use a single microbenchmark size.
- Explain the asymptotic costs and the actual kernel/runtime bottlenecks.
- Use CPU and CUDA profiler evidence when available. Why do some vectorized
  rewrites reduce Python overhead yet increase GPU memory traffic?
- Compare a change that improves large batch throughput but hurts small-batch
  latency. Discuss the Pareto frontier of memory, accuracy, throughput and
  time to first token.
- Profile autoregressive generation and inspect cache semantics and growth.
- Run adversarial/differential correctness tests with small reference inputs.

### Part 4 — Final adversarial review

A reviewer asks whether your speedups are real. Provide baseline and optimized
runtime medians, warmup policy, device, PyTorch version, memory measurements,
end-to-end throughput, and precise correctness tolerance. Show both a win and
a regression. Explain whether a micro-optimization matters at system scale.

## Execution

Requires Python 3.10+, PyTorch 2.x, and pytest. CPU-only execution is valid;
CUDA-specific profiling, AMP and synchronization exercises require CUDA.

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m benchmarks.bench_train --device cpu --batch 4 --seq 16 --steps 2
python -m benchmarks.bench_attention --device cpu --seq 32 --batch 2
python -m benchmarks.compare_baseline --seq 16 --batch 2
python -m benchmarks.profile_train --device cpu --steps 2
```

On supported GPU hardware repeat with `--device cuda`, real warmups and scales
32/64/128/256 in sequence length and 4/8/16/32 in batch. Do not blindly run
an enormous input on an unknown GPU. On CPU, start smaller and scale only if
runtime permits.

Individual scenario tests:

```bash
python -m pytest tests/test_scenario_attention_logits.py -vv
python -m pytest tests/test_integration_attention.py -vv
python -m pytest tests/test_pipeline_end_to_end.py -vv
```

## Important design distinctions

- Existing slow baseline functions can be correct; **performance bugs are not
  always correctness bugs**. Do not weaken tests to achieve speedups.
- Smoke tests do not prove that a fused kernel, mixed precision, graph compile
  or cache strategy is correct. Add comparison tests for logits, loss,
  gradients, optimizer state and generation.
- `torch.no_grad()` and `torch.inference_mode()` are not interchangeable
  with training execution. Scope them only to semantics that permit it.
- Never compare async CUDA timings without synchronization.
- `torch.compile` startup overhead must be amortized and reported.
- Partial final microbatch and padded-sequence loss reductions need explicit
  denominator choices. Explain each observed metric.
- Do not conflate allocated and reserved GPU memory.
- Not all 42 scenarios are expected in one interview; this is an intentionally
  extreme bank of related workloads in the same repository.

## Scoring rubric (100 points)

- 20: codebase navigation and accurate computational graph explanation
- 20: high-impact bottleneck diagnosis supported by measurements
- 20: semantic preservation and independent regression/differential tests
- 20: measured performance and memory improvements on relevant workloads
- 10: realistic scaling, precision, distributed and compile trade-offs
- 10: concise communication and prioritization under 45-minute constraints

The repository contains no answer key and no requirements to use paid APIs.
