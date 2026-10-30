"""Big layer 2: fusion blocks ("搭积木") and the basic TTSN assembly.

Building-block style: every fusion block has the same interface

    block(a, b) -> fused embedding

so blocks can be stacked — fuse one pair, feed the result into the next
block — and the chain ends in one final embedding.

Simplest wiring (BasicTTSN):

    user_emb  = UserTowerEncoder(user_info)          # [B, 128]
    video_emb = VideoTowerEncoder(video_info)        # [B, 128]
    inter_emb = InteractionEncoder(interactive_info)  # [B, 112]

    final_user  = ConcatMLPFusion(user_emb,  inter_emb)   # [B, 128]
    final_video = ConcatMLPFusion(video_emb, inter_emb)   # [B, 128]

Other fusion options — TODO(fusion-blocks), same interface, swap in later:
    - gated: b decides how much of each channel of a passes (sigmoid gate)
    - sum / mean pooling after projecting to the same width
    - bilinear: a^T W b per output channel
    - cross-attention: a attends over b's feature groups
    - FiLM: b produces scale/shift applied to a
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .encoding import InteractionEncoder, UserTowerEncoder, VideoTowerEncoder
from .features import UserInfo, UserVideoInteractiveInfo, VideoInfo
from .model import TTSNConfig, TwoTowerModel


class ConcatMLPFusion(nn.Module):
    """Simplest fusion: concat(a, b) -> MLP -> out_dim."""

    def __init__(self, a_dim: int, b_dim: int, out_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(a_dim + b_dim),
            nn.Linear(a_dim + b_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([a, b], dim=-1))


class BasicTTSN(TwoTowerModel):
    """Basic TTSN: tower encodings fused with the interaction encoding.

    forward(user_info, video_info, interactive_info):
        user side:  user_emb  (+) inter_emb -> final MLP -> final user emb
        video side: video_emb (+) inter_emb -> final MLP -> final video emb

    Note: mixing interaction features into both sides means embeddings can
    not be precomputed per-user/per-video independently — fine for ranking;
    for ANN retrieval serving the interaction input would be zeroed/defaulted.
    """

    def __init__(
        self,
        config: Optional[TTSNConfig] = None,
        emb_dim: int = 16,
        tower_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__(config)
        out = self.config.output_dim
        self.user_encoder = UserTowerEncoder(
            emb_dim=emb_dim, output_dim=out, hidden_dim=tower_hidden_dim)
        self.video_encoder = VideoTowerEncoder(
            emb_dim=emb_dim, output_dim=out, hidden_dim=tower_hidden_dim)
        self.interaction_encoder = InteractionEncoder(emb_dim=emb_dim)

        inter_dim = self.interaction_encoder.output_dim
        self.user_fusion = ConcatMLPFusion(out, inter_dim, out, fusion_hidden_dim)
        self.video_fusion = ConcatMLPFusion(out, inter_dim, out, fusion_hidden_dim)

    def forward(
        self,
        user_info: UserInfo,
        video_info: VideoInfo,
        interactive_info: Optional[UserVideoInteractiveInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_encoder(user_info)        # [B, out]
        video_emb = self.video_encoder(video_info)     # [B, out]
        inter_emb = self.interaction_encoder(interactive_info)  # [B, 7*emb_dim]

        final_user = self.user_fusion(user_emb, inter_emb)    # [B, out]
        final_video = self.video_fusion(video_emb, inter_emb)  # [B, out]
        return final_user, final_video
