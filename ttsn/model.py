"""Two-tower (TTSN) model skeleton.

``TwoTowerModel`` subclasses ``torch.nn.Module`` and fixes one contract:

    forward(user_info, video_info, interactive_info) -> (user_emb, video_emb)

The inputs are the three info objects directly (UserInfo, VideoInfo,
UserVideoInteractiveInfo); how to split them into towers / encode features
is entirely up to the subclass. Both outputs are [B, output_dim] (default
128, configurable). ``forward`` is intentionally left empty here — this
layer only defines the interface, the loss wiring and the train/infer
autograd policy:

- training: ``training_step`` runs forward + loss with autograd enabled.
- inference: ``infer`` / ``score`` run under ``torch.inference_mode``
  (no graph, no grad).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .features import UserInfo, UserVideoInteractiveInfo, VideoInfo
from .loss import in_batch_negative_ce_loss


@dataclass(kw_only=True)
class TTSNConfig:
    output_dim: int = 128          # final embedding dim of both towers
    # softmax temperature: logits = cosine / temperature, so gradients scale
    # ~1/temperature. 0.05 suits huge in-batch-negative batches (thousands);
    # with small batches it makes gradients explode (measured: temp 0.05 ->
    # grad norm ~90, temp 0.2 -> ~15, temp 1.0 -> ~2.7). Default 0.2.
    temperature: float = 0.2
    normalize: bool = True         # L2-normalize embeddings before use


class TwoTowerModel(nn.Module):
    """Base two-tower model; ``forward`` is left for subclasses.

    TODO(forward): implement the basic TTSN — id-embedding tables,
    dense/categorical feature processing, historical_clicks sequence
    encoder, scenario conditioning, one MLP tower per side.
    """

    def __init__(self, config: Optional[TTSNConfig] = None) -> None:
        super().__init__()
        self.config = config or TTSNConfig()

    # ------------------------------------------------------------------
    # Forward: left empty on purpose; subclasses implement the full
    # two-tower computation from the three info objects.
    # ------------------------------------------------------------------
    def forward(
        self,
        user_info: UserInfo,
        video_info: VideoInfo,
        interactive_info: Optional[UserVideoInteractiveInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (user_emb, video_emb), both [B, output_dim]."""
        raise NotImplementedError("subclass must implement forward")

    # ------------------------------------------------------------------
    # Output contract: shape check + optional L2 normalization.
    # ------------------------------------------------------------------
    def _finalize(
        self, user_emb: torch.Tensor, video_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_dim(user_emb, "user")
        self._check_dim(video_emb, "video")
        if self.config.normalize:
            user_emb = F.normalize(user_emb, dim=-1)
            video_emb = F.normalize(video_emb, dim=-1)
        return user_emb, video_emb

    def _check_dim(self, emb: torch.Tensor, side: str) -> None:
        if emb.dim() != 2 or emb.size(-1) != self.config.output_dim:
            raise ValueError(
                f"{side} embedding must be [B, {self.config.output_dim}], "
                f"got {tuple(emb.shape)}"
            )

    # ------------------------------------------------------------------
    # Loss: in-batch negatives + per-sample weights. CE for now; pluggable.
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        user_emb: torch.Tensor,
        video_emb: torch.Tensor,
        user_ids=None,
        video_ids=None,
        positive_mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return in_batch_negative_ce_loss(
            user_emb,
            video_emb,
            user_ids=user_ids,
            video_ids=video_ids,
            positive_mask=positive_mask,
            sample_weights=sample_weights,
            temperature=self.config.temperature,
        )

    def training_step(
        self,
        user_info: UserInfo,
        video_info: VideoInfo,
        interactive_info: Optional[UserVideoInteractiveInfo] = None,
        sample_weights: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward + loss with autograd enabled (even if caller disabled it).

        Real user/video ids go to the loss; random batches may repeat users
        or docs, and the label mask reflects the actual click facts.
        """
        with torch.enable_grad():
            user_emb, video_emb = self(user_info, video_info, interactive_info)
            user_emb, video_emb = self._finalize(user_emb, video_emb)
            return self.compute_loss(
                user_emb,
                video_emb,
                user_ids=user_info.user_id,
                video_ids=video_info.video_id,
                positive_mask=positive_mask,
                sample_weights=sample_weights,
            )

    # ------------------------------------------------------------------
    # Inference: no autograd, eval mode.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def infer(
        self,
        user_info: UserInfo,
        video_info: VideoInfo,
        interactive_info: Optional[UserVideoInteractiveInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(user_emb, video_emb) without autograd, in eval mode."""
        self.eval()
        user_emb, video_emb = self(user_info, video_info, interactive_info)
        return self._finalize(user_emb, video_emb)

    @torch.inference_mode()
    def score(
        self,
        user_info: UserInfo,
        video_info: VideoInfo,
        interactive_info: Optional[UserVideoInteractiveInfo] = None,
    ) -> torch.Tensor:
        """Pairwise relevance score [B] = cosine(user_i, video_i)."""
        user_emb, video_emb = self.infer(user_info, video_info, interactive_info)
        return (user_emb * video_emb).sum(dim=-1)
