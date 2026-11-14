"""CPU source-line timing + CUDA operation report, run in separate passes.

The CPU line tracer does not depend on torch.profiler event.stack and therefore
works when PyTorch's Python stack attribution is unavailable. CPU measurements
are instrumented wall-clock elapsed time attributed to executing Python source
lines; they include any CPU-GPU waits, Python/C++ dispatch and tracing overhead.
They are NOT isolated CPU compute cycles. GPU events are collected separately.

Run from the round06_extreme_perf directory:
  python -m benchmarks.profile_source_lines --device cuda --steps 2
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from perfkit.infra.config import ModelConfig
from perfkit.model.decoder import ResearchDecoder
from perfkit.data.corpus import synthetic_corpus
from perfkit.train.optimizers import make_optimizer
from perfkit.train.engine import ResearchTrainer

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "perfkit"


def _is_project_file(filename):
    try:
        return Path(filename).resolve().is_relative_to(PACKAGE)
    except (OSError, ValueError):
        return False


class CPULineTracer:
    """Trace the last executed perfkit source line, including nested calls.

    This measures elapsed CPU-side *wall time*, not CPU instruction execution.
    Time in external Python/C++ calls is attributed to the calling perfkit line.
    """
    def __init__(self):
        self.counts = defaultdict(int)
        self.elapsed_ns = defaultdict(int)
        self._current = None
        self._last_ns = None
        self._previous = None

    def _key(self, frame):
        return (Path(frame.f_code.co_filename).resolve().relative_to(PROJECT).as_posix(),
                frame.f_lineno)

    def _trace(self, frame, event, arg):
        if event == "call":
            if _is_project_file(frame.f_code.co_filename):
                return self._trace
            return None
        now = time.perf_counter_ns()
        if self._last_ns is not None and self._current is not None:
            self.elapsed_ns[self._current] += now - self._last_ns
        self._last_ns = now
        if event == "line":
            self._current = self._key(frame)
            self.counts[self._current] += 1
        elif event in ("return", "exception"):
            # Attribute resumed work to the caller's current source line.
            parent = frame.f_back
            while parent is not None and not _is_project_file(parent.f_code.co_filename):
                parent = parent.f_back
            self._current = self._key(parent) if parent else None
        return self._trace

    def __enter__(self):
        self._previous = sys.gettrace()
        self._last_ns = time.perf_counter_ns()
        sys.settrace(self._trace)
        return self

    def __exit__(self, *_):
        sys.settrace(self._previous)
        self._last_ns = None
        self._current = None

    def rows(self):
        return sorted(((f"{name}:{line}", self.counts[(name, line)],
                        self.elapsed_ns[(name, line)] / 1e6)
                       for name, line in self.counts),
                      key=lambda row: row[2], reverse=True)


def _model(device, backend):
    cfg = ModelConfig(vocab_size=128, hidden_size=32, num_heads=4,
                      num_layers=1, intermediate_size=64,
                      max_position_embeddings=80)
    model = ResearchDecoder(cfg).to(device)
    records = synthetic_corpus(32, seed=16, vocab=128,
                               min_length=24, max_length=32)
    return ResearchTrainer(model, make_optimizer(model), device,
                           backend=backend), records


def _source_location(frames):
    for entry in frames:
        normalized = str(entry).replace("\\", "/")
        if "/perfkit/" in normalized or normalized.startswith("perfkit/"):
            return normalized
    return None


def _write_csv(path, columns, rows):
    if not path:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print("Saved:", path)


def profile_source_lines(device="cuda", steps=2, backend="research",
                         csv_path="source_line_cpu_report.csv",
                         gpu_csv_path="source_line_gpu_triggers.csv",
                         trace="source_line_profile.json"):
    dev = torch.device(device)
    if steps < 1:
        raise ValueError("steps must be positive")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # Pass 1: source-line timing, without torch.profiler. Independent run.
    torch.manual_seed(123)
    trainer, records = _model(dev, backend)
    with CPULineTracer() as tracer:
        for _ in range(steps):
            trainer.one_step(records[:4], microbatch=2)
    cpu_rows = tracer.rows()
    if not cpu_rows:
        raise RuntimeError("CPU line tracer produced no perfkit source events")
    print("\nCPU SOURCE-LINE REPORT (instrumented elapsed wall time, not pure CPU compute)")
    print(f'{"Source location":77} {"Calls":>7} {"Elapsed ms":>12}')
    for loc, count, ms in cpu_rows[:45]:
        print(f"{loc[-77:]:77} {count:7d} {ms:12.3f}")
    _write_csv(csv_path, ["source_line", "calls", "elapsed_wall_ms"], cpu_rows)

    # Pass 2: CUDA operator attribution, with optional source stacks.
    torch.manual_seed(123)
    trainer2, records2 = _model(dev, backend)
    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_stack=True, record_shapes=False,
                 profile_memory=False, acc_events=True) as prof:
        for _ in range(steps):
            with record_function("research_train_step"):
                trainer2.one_step(records2[:4], microbatch=2)
            prof.step()

    cuda_rows = defaultdict(lambda: [0, 0.0])
    mapped = 0
    for event in prof.events():
        us = float(getattr(event, "self_device_time_total", 0.0))
        if us <= 0:
            continue
        loc = _source_location(getattr(event, "stack", None) or [])
        if loc is not None:
            mapped += 1
        # Retain unattributed CUDA operations instead of silently dropping them.
        key = (loc or "<source unavailable>", str(event.name))
        cuda_rows[key][0] += 1
        cuda_rows[key][1] += us / 1000
    gpu_rows = sorted(((loc, op, count, ms) for (loc, op), (count, ms)
                       in cuda_rows.items()), key=lambda row: row[3], reverse=True)
    print("\nGPU OPERATOR REPORT (kernel/ATen time, Python source if supplied)")
    print(f'{"Location":60} {"Operator":33} {"Calls":>7} {"CUDA ms":>10}')
    for loc, op, count, ms in gpu_rows[:35]:
        print(f"{loc[-60:]:60} {op[:33]:33} {count:7d} {ms:10.3f}")
    _write_csv(gpu_csv_path,
               ["source_location", "operator", "calls", "self_cuda_ms"], gpu_rows)
    if dev.type == "cuda" and not gpu_rows:
        print("Warning: CUDA activity not available in profiler output")
    if dev.type == "cuda" and mapped == 0:
        print("Note: this PyTorch build supplies no CUDA source stacks; "
              "operators remain listed as <source unavailable>.")
    if trace:
        prof.export_chrome_trace(trace)
        print("Trace:", trace)
    return {"cpu_lines": cpu_rows, "gpu_ops": gpu_rows, "profiler": prof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--backend", choices=["research", "sdpa"], default="research")
    ap.add_argument("--csv", default="source_line_cpu_report.csv")
    ap.add_argument("--gpu-csv", default="source_line_gpu_triggers.csv")
    ap.add_argument("--trace", default="source_line_profile.json")
    a = ap.parse_args()
    profile_source_lines(a.device, a.steps, a.backend, a.csv, a.gpu_csv, a.trace)


if __name__ == "__main__":
    main()