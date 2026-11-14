"""CPU source-line attribution with GPU operator launch-site context.

PyTorch profiler times are instrumentation-affected and do not represent
hardware-only CPU computation. Self CPU excludes nested profiled children.
Run from round06_extreme_perf:
  python -m benchmarks.profile_source_lines --device cuda --steps 2
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from benchmarks.profile_functions import labeled_functions
from perfkit.infra.config import ModelConfig
from perfkit.model.decoder import ResearchDecoder
from perfkit.data.corpus import synthetic_corpus
from perfkit.train.optimizers import make_optimizer
from perfkit.train.engine import ResearchTrainer


def _source_line(frames):
    # Prefer the deepest project frame, excluding profiler helper code.
    for frame in frames:
        if "perfkit/" in frame.replace("\\", "/"):
            return frame.strip().replace("/content/round06_training_repo/round06_extreme_perf/", "")
    return None


def profile_source_lines(device="cuda", steps=2, backend="research",
                         csv_path="source_line_cpu_report.csv",
                         gpu_csv_path="source_line_gpu_triggers.csv",
                         trace="source_line_profile.json"):
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    cfg = ModelConfig(vocab_size=128, hidden_size=32, num_heads=4,
                      num_layers=1, intermediate_size=64,
                      max_position_embeddings=80)
    model = ResearchDecoder(cfg).to(dev)
    optimizer = make_optimizer(model)
    trainer = ResearchTrainer(model, optimizer, device=dev, backend=backend)
    records = synthetic_corpus(32, seed=16, vocab=128,
                               min_length=24, max_length=32)
    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with labeled_functions(model, optimizer):
        with profile(activities=activities, with_stack=True,
                     record_shapes=False, profile_memory=False) as prof:
            for _ in range(steps):
                with record_function("source_profile_train_step"):
                    trainer.one_step(records[:4], microbatch=2)
                prof.step()

    by_cpu = defaultdict(lambda: [0, 0.0, 0.0])
    by_gpu = defaultdict(lambda: [0, 0.0])
    matched = 0
    for event in prof.events():
        location = _source_line(event.stack or [])
        if location is None:
            continue
        matched += 1
        cpu = by_cpu[location]
        cpu[0] += 1
        cpu[1] += float(event.self_cpu_time_total) / 1000
        cpu[2] += float(event.cpu_time_total) / 1000
        cuda_us = float(getattr(event, "self_device_time_total", 0.0))
        if cuda_us > 0:
            gpu = by_gpu[(location, event.name)]
            gpu[0] += 1
            gpu[1] += cuda_us / 1000

    cpu_rows = sorted(((loc, *data) for loc, data in by_cpu.items()),
                      key=lambda x: x[2], reverse=True)
    gpu_rows = sorted(((loc, op, *data)
                       for (loc, op), data in by_gpu.items()),
                      key=lambda x: x[3], reverse=True)

    print("\nCPU source-line report (5+ columns; timing includes profiling overhead)")
    print(f'{"Source location":85} {"Calls":>7} {"Self CPU ms":>13} {"CPU total ms":>14}')
    for loc, n, self_ms, total_ms in cpu_rows[:35]:
        print(f"{loc[-85:]:85} {n:7d} {self_ms:13.3f} {total_ms:14.3f}")

    print("\nGPU operations mapped to Python source lines (not exclusive line GPU time)")
    print(f'{"Source location":72} {"Operation":31} {"Calls":>7} {"CUDA ms":>10}')
    for loc, op, n, ms in gpu_rows[:35]:
        print(f"{loc[-72:]:72} {op[:31]:31} {n:7d} {ms:10.3f}")

    for filename, header, rows in (
        (csv_path, ["location", "calls", "self_cpu_ms", "cpu_total_ms"], cpu_rows),
        (gpu_csv_path, ["location", "operator", "calls", "self_cuda_ms"], gpu_rows),
    ):
        if filename:
            with open(filename, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerows(rows)
            print("Saved:", filename)
    if trace:
        prof.export_chrome_trace(trace)
        print("Trace:", trace)
    if not matched:
        print("WARNING: no perfkit/ Python source frames were available.")
        print("Raw stack example:", next((e.stack for e in prof.events() if e.stack), []))
    return prof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--backend", choices=["research", "sdpa"], default="research")
    parser.add_argument("--csv", default="source_line_cpu_report.csv")
    parser.add_argument("--gpu-csv", default="source_line_gpu_triggers.csv")
    parser.add_argument("--trace", default="source_line_profile.json")
    args = parser.parse_args()
    profile_source_lines(args.device, args.steps, args.backend,
                         args.csv, args.gpu_csv, args.trace)


if __name__ == "__main__":
    main()
