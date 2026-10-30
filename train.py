"""TTSN training entry point.

Training protocol:
- data: 30 days of mock interaction logs; the LAST 2 days are the test set
  (temporal split — never random, to avoid future leakage)
- the train API is per-batch: each call feeds one batch (default 1000 rows,
  which is also the in-batch-negative pool), computes the weighted CE loss,
  backprops, and takes one optimizer step
- one epoch = every training batch walked exactly once (reshuffled each epoch)
- eval after each epoch: weighted CE + in-batch top-1 accuracy on the
  held-out days (no grad)

Run:  python train.py [--epochs 3] [--batch-size 1000] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import time

import torch

from ttsn.data import MockDataConfig, generate_mock_data, iter_batches, make_batch, temporal_split
from ttsn.fusion import BasicTTSN
from ttsn.model import TTSNConfig


def train_one_batch(model, optimizer, batch, max_grad_norm: float = 10.0) -> float:
    """The per-batch train API: one batch in, one gradient step out."""
    user, video, inter, weights = batch
    optimizer.zero_grad()
    loss = model.training_step(user, video, inter, sample_weights=weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, test_idx, batch_size: int) -> tuple[float, float]:
    """Weighted CE + in-batch top-1 accuracy over the held-out days."""
    model.eval()
    total_loss, total_correct, total_rows, num_batches = 0.0, 0, 0, 0
    for idx in iter_batches(test_idx, batch_size, shuffle=False):
        user, video, inter, weights = make_batch(data, idx)
        user_emb, video_emb = model.infer(user, video, inter)
        loss = model.compute_loss(
            user_emb, video_emb,
            user_ids=user.user_id, video_ids=video.video_id,
            sample_weights=weights,
        )
        total_loss += loss.item()
        num_batches += 1
        # top-1: does user i rank its own clicked video highest in the batch
        logits = user_emb @ video_emb.t()
        total_correct += (logits.argmax(1) == torch.arange(len(idx))).sum().item()
        total_rows += len(idx)
    model.train()
    return total_loss / max(num_batches, 1), total_correct / max(total_rows, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the basic TTSN model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interactions-per-day", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = MockDataConfig(
        days=args.days, interactions_per_day=args.interactions_per_day, seed=args.seed
    )
    data = generate_mock_data(cfg)
    train_idx, test_idx = temporal_split(data, cfg)
    print(f"data: {cfg.days} days, {len(train_idx)} train rows "
          f"(days 0-{cfg.days - cfg.test_days - 1}), "
          f"{len(test_idx)} test rows (last {cfg.test_days} days)")

    model = BasicTTSN(TTSNConfig(temperature=args.temperature))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    shuffler = torch.Generator().manual_seed(args.seed)

    test_loss, test_acc = evaluate(model, data, test_idx, args.batch_size)
    print(f"before training: test_loss={test_loss:.4f} test_top1={test_acc:.3f}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        epoch_loss, num_batches = 0.0, 0
        # one epoch = every training batch exactly once
        for idx in iter_batches(train_idx, args.batch_size, shuffle=True, generator=shuffler):
            batch = make_batch(data, idx)
            epoch_loss += train_one_batch(model, optimizer, batch)
            num_batches += 1
        test_loss, test_acc = evaluate(model, data, test_idx, args.batch_size)
        print(f"epoch {epoch}: train_loss={epoch_loss / num_batches:.4f} "
              f"test_loss={test_loss:.4f} test_top1={test_acc:.3f} "
              f"({num_batches} batches, {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
