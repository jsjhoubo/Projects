"""Loss functions for TTSN training.

In-batch negative sampling for one-directional user -> video retrieval
(row-wise only; no symmetric video -> user term like CLIP). For a batch of
B (user, video) clicked pairs, row i of the B x B similarity matrix scores
user i against every video in the batch.

Batches are randomly sampled — duplicate users and duplicate docs both
happen. The label matrix just reflects the actual click facts we know:
user i clicked d_i, so video_j is a true positive for row i when

    positive[i, j] = (user_ids[j] == user_ids[i])     # same user's click
                   | (video_ids[j] == video_ids[i])   # same doc as d_i

Loss pipeline, written out explicitly (standard cross-entropy):

    logits  = U @ V.T / temperature          # [B, B] similarities
    logits -= logits.max(dim=1)              # numerical stability shift
    probs   = softmax(logits, dim=1)         # [B, B] probability matrix
    CE_i    = -sum_j label_ij * log(probs_ij)

i.e. every true positive contributes its own -log p term: a user with
positives d1, d2 gets -(log p(u,d1) + log p(u,d2)); when every row is a
distinct user and doc this reduces exactly to ordinary diagonal-label CE.

TODO(logq-correction): in-batch negatives oversample popular videos as
negatives (they appear in more batches). Correction: subtract log Q(video_j)
from logits[i, j] before the softmax, where Q is each video's probability of
appearing in a batch (streaming frequency estimate). See Yi et al. 2019.

Worked example (B=3), rows = clicked pairs (u1,a), (u1,b), (u2,c):

    user_ids  = [u1, u1, u2]
    video_ids = [a,  b,  c ]

    label mask (same user OR same doc; diagonal always True):
        row0 (u1): [1, 1, 0]   # u1 clicked both a (row0) and b (row1)
        row1 (u1): [1, 1, 0]
        row2 (u2): [0, 0, 1]

    say logits = U @ V.T / temperature came out as:
        [[2, 1, 0],
         [1, 3, 1],
         [0, 1, 2]]

    after per-row max shift (row0 -2, row1 -3, row2 -2):
        [[ 0, -1, -2],
         [-2,  0, -2],
         [-2, -1,  0]]

    probs = softmax rows (exp / row-sum):
        [[0.665, 0.245, 0.090],
         [0.107, 0.787, 0.107],
         [0.090, 0.245, 0.665]]

    log(probs):
        [[-0.408, -1.408, -2.408],
         [-2.240, -0.240, -2.240],
         [-2.408, -1.408, -0.408]]

    labels * log(probs), then -sum per row:
        CE_0 = -(-0.408 + -1.408) = 1.816
        CE_1 = -(-2.240 + -0.240) = 2.480
        CE_2 = -(-0.408)          = 0.408

    sample_weights = [1, 2, 1]:
        loss = (1*1.816 + 2*2.480 + 1*0.408) / (1+2+1) ~= 1.796
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F


def build_positive_mask(
    user_ids: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
    video_ids: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """[B, B] bool mask of true positives for user -> video retrieval.

    Batches are random (no dedup), so the mask encodes the actual click
    facts: video_j is positive for row i when the same user produced rows
    i and j (same user's click) OR video_j is the same doc as d_i:

        mask[i, j] = same_user[i, j] | same_video[i, j]

    Works with id tensors or lists of hashables. The diagonal is always True.
    """
    if user_ids is None and video_ids is None:
        raise ValueError("need user_ids and/or video_ids")
    mask: Optional[torch.Tensor] = None
    if user_ids is not None:
        mask = _eq_matrix(user_ids)
    if video_ids is not None:
        same_video = _eq_matrix(video_ids)
        mask = same_video if mask is None else (mask | same_video)
    return mask.to(device) if device is not None else mask


def _eq_matrix(ids: Union[torch.Tensor, Sequence[Any]]) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        flat = ids.reshape(-1)
        return flat.unsqueeze(0) == flat.unsqueeze(1)
    return torch.tensor([[a == b for b in ids] for a in ids], dtype=torch.bool)


def in_batch_negative_ce_loss(
    user_emb: torch.Tensor,
    video_emb: torch.Tensor,
    user_ids: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
    video_ids: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
    positive_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Weighted multi-positive softmax CE over in-batch negatives.

    One-directional user -> video retrieval: softmax over videos per user
    row only (no symmetric video -> user term). Standard CE with a label
    matrix: CE_i = -sum_j label_ij * log p(u_i, d_j) — a user with two
    positives d1, d2 in the batch gets -(log p(u,d1) + log p(u,d2)).

    Args:
        user_emb:  [B, D] user tower output (assumed L2-normalized).
        video_emb: [B, D] video tower output, row i is a doc user i clicked.
        user_ids: optional [B] real user ids; rows of the same user share all
            of that user's clicked docs as positives.
        video_ids: optional [B] real video ids; duplicate docs in the random
            batch are positives for every row whose user clicked that doc.
        positive_mask: optional [B, B] bool/0-1 matrix of true labels,
            overriding ``user_ids``/``video_ids``. mask[i, j] = video_j is
            positive for user i. Each row must contain at least one positive.
        sample_weights: optional [B] non-negative per-sample weights.
        temperature: softmax temperature for the similarity logits.

    When no label inputs are given, falls back to diagonal labels (assumes
    all batch rows are distinct users and docs).

    Returns:
        Scalar loss (weighted mean of per-sample CE).
    """
    if user_emb.shape != video_emb.shape:
        raise ValueError(
            f"shape mismatch: user {tuple(user_emb.shape)} vs "
            f"video {tuple(video_emb.shape)}"
        )
    batch_size = user_emb.size(0)
    device = user_emb.device

    # [B, B] true-label matrix from real labels
    if positive_mask is not None:
        mask = positive_mask.to(device=device, dtype=torch.bool)
    elif user_ids is not None or video_ids is not None:
        mask = build_positive_mask(
            user_ids=user_ids, video_ids=video_ids, device=device
        )
    else:
        mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    if mask.shape != (batch_size, batch_size):
        raise ValueError(
            f"positive mask must be [B, B]=[{batch_size}, {batch_size}], "
            f"got {tuple(mask.shape)}"
        )
    if not mask.any(dim=1).all():
        raise ValueError("every row needs at least one positive label")

    # ---- step 1: similarity logits ----
    # [B, B]: logits[i, j] = <user_i, video_j> / temperature
    logits = user_emb @ video_emb.t() / temperature

    # numerical stability: subtract per-row max so exp() never overflows;
    # softmax is shift-invariant so the result is unchanged
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # ---- step 2: row-wise softmax probability matrix ----
    # [B, B]: probs[i, j] = softmax over row i = p(u_i clicks d_j)
    probs = F.softmax(logits, dim=1)

    # ---- step 3: standard CE:  CE_i = -sum_j label_ij * log p_ij ----
    # clamp before log so an underflowed prob (0.0) doesn't give log(0)=-inf
    log_probs = torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny))
    labels = mask.to(log_probs.dtype)             # [B, B] label matrix
    per_sample = -(labels * log_probs).sum(dim=1)  # [B]

    if sample_weights is None:
        return per_sample.mean()

    if sample_weights.shape != (batch_size,):
        raise ValueError(
            f"sample_weights must be [B]={batch_size}, got "
            f"{tuple(sample_weights.shape)}"
        )
    weights = sample_weights.to(per_sample.dtype)
    denom = weights.sum().clamp_min(torch.finfo(per_sample.dtype).eps)
    return (per_sample * weights).sum() / denom
