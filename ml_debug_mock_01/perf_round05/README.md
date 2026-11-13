# Round 05 — Extreme PyTorch Performance Engineering Mock

**One integrated exercise, intentionally harder than the Resolution interview.**
Original mock—not official/leaked material from Resolution, OpenAI, or Anthropic.

## Verified Resolution interview format

45-minute ML-style performance optimization review. Interviewer supplies
poorly written experimental **PyTorch Colab** code; expects explanation and
improvements, with **no AI and no code-writing required**. Hugging Face only
marginally relevant. This mock adds demanding implementation and benchmarking
work, but the first 45 minutes should still be practiced without AI.

## Scenario

A research team's small causal LM trains, validates, and decodes correctly.
Yet GPU utilization is poor, scaling batch/sequence length is erratic,
validation blocks training, and memory usage spikes. You inherit one codebase
and must deliver end-to-end optimization *without changing model semantics*.

Code: `perf_round05/research_model.py` (also packaged as Colab
`perf_round05/extreme_perf_mock.ipynb`).
Tests: `perf_round05/test_perf_contracts.py`.

## Unified interview assignment — difficult version

**Phase 1: 45-minute blind code review (Resolution simulation).**
Without editing or using AI, annotate the notebook:
- Trace input -> forward -> causal attention -> loss -> backward -> optimizer.
- Identify bottlenecks and rank them by estimated end-to-end impact.
- Distinguish GPU kernel-launch overhead, synchronization, data copies,
  autograd memory, inefficient batching, validation, and autoregressive decode.
- Estimate attention and activations as functions of B, T, H, head_dim, vocab.
- Propose an optimization, a measurement, a correctness check and a risk for
  each hotspot. Evaluate possible regressions from mixed precision or SDPA.
- Explain why CPU-only timings may not predict GPU timings.

**Phase 2: hands-on performance rescue (OpenAI-style).**
Implement one or several optimized variants **in a separate module**.
Optimize throughput, p95 step latency, and memory on CPU/GPU. Include
`torch.profiler` evidence; synchronize CUDA correctly. Handle batch sizes
8/16/32, sequence lengths 32/64/128, and skewed sequence lengths.
Show warm-up/cold-start effects, and measure sustained throughput.

**Phase 3: hard algorithmic/runtime extensions.**
Address and validate:
- Correct causal mask/batching under PyTorch SDPA or fused attention; whether
  gradients match the reference within floating-point tolerance.
- Gradient accumulation semantics, optimizer clipping, missing grads,
  precision changes, activation checkpointing, and DDP `no_sync`.
- Autoregressive generation complexity, KV-cache invariants and positional
  offsets under context-window truncation.
- Reproducibility with stochastic dropout, deterministic seeds, and
  numerical tolerance—not bit-for-bit equivalence.
- Contrast O(B H T^2 d) computation with quadratic attention memory, memory
  bandwidth, kernel launch count, and host-device sync costs.
- Push 2–4x longer sequences and discuss where the bottleneck changes.
- Find an optimization that helps throughput but hurts small-batch latency.
- Determine whether `torch.compile` and fused optimizers help after warmup,
  and assess compilation/graph-break risks.

**Phase 4: Anthropic-style correctness/performance discipline.**
No self-reported "seems faster": create an independent correctness oracle and
report exact baseline/optimized outputs, gradient diffs, elapsed time,
peak allocated memory and distribution of step latencies. Avoid flaky
machine-specific time assertions. Explain Pareto tradeoffs and when a
faster kernel does not accelerate the whole program. This is inspired by
performance-engineering rigor, not the official Anthropic problem itself.

## Mandatory regression gates

```powershell
python -m pytest perf_round05/test_perf_contracts.py -vv
python -m perf_round05.research_model --seq 48 --batch 16 --repeats 3
python -m perf_round05.research_model --seq 128 --batch 16 --repeats 3
```

Tests are smoke contracts, *not* sufficient proof for a new implementation.
For every optimization add differential tests on logits, loss, gradients,
optimizer update, and greedy tokens. Beware PyTorch version/backend differences.
No answer key, specific solution locations, or official employer questions
are included.

## Original official Anthropic take-home

https://github.com/anthropics/original_performance_takehome

The official task optimizes a simulated accelerator's instruction/kernel
execution; it is a different programming environment. Learn from its strict
performance-vs-correctness evaluation model, but do not mistake it for a
Resolution PyTorch Colab question.
