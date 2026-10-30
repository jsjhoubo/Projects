"""First encoding layer ("big layer 1") for the TTSN model.

Encoding rules (see feature_config.py for the per-feature spec):

- sparse features: stable hash into N buckets -> one single embedding layer
  per feature (user_id/video_id hash 10000, everything else hash 1000)
- dense features: normalize into [0, 1] first, then bucketize (<= 100
  buckets), then embed
- user_location: geo encoding — latitude and longitude bucketized separately
- video title/tags: embeddings already produced by a known upstream model,
  linearly projected to the shared embedding width
- interaction windows: per event type (click / like / comment / dislike) one
  encoding input — the (1d, 1w, 1m) binary flags form a 3-bit pattern [0..7]
  looked up in that event's own embedding table

Tower assembly:
- UserTowerEncoder / VideoTowerEncoder: concat all feature embeddings
  -> MLP -> output_dim (128)
- InteractionEncoder: concat only (user_id, video_id, 4 event patterns,
  watch_time) — the next big layer on top is TBD.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .features import UserInfo, UserVideoInteractiveInfo, VideoInfo

MAX_DENSE_BUCKETS = 100


def stable_hash_bucket(value: Any, num_buckets: int) -> int:
    """Deterministic (process-independent) hash of any value into a bucket."""
    digest = hashlib.md5(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % num_buckets


def hash_to_buckets(
    values: Union[torch.Tensor, Sequence[Any]],
    num_buckets: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """[B] long tensor of bucket indices for raw ids (tensor or hashables)."""
    if isinstance(values, torch.Tensor):
        return (values.reshape(-1).long() % num_buckets).to(device)
    idx = [stable_hash_bucket(v, num_buckets) for v in values]
    return torch.tensor(idx, dtype=torch.long, device=device)


class HashEmbedding(nn.Module):
    """sparse -> hash N buckets -> single embedding layer."""

    def __init__(self, num_buckets: int, emb_dim: int) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, emb_dim)

    def forward(self, raw_values: Union[torch.Tensor, Sequence[Any]]) -> torch.Tensor:
        idx = hash_to_buckets(raw_values, self.num_buckets, self.emb.weight.device)
        return self.emb(idx)  # [B, emb_dim]


class DenseBucketEmbedding(nn.Module):
    """dense -> normalize to [0,1] -> bucketize (<=100) -> embedding."""

    def __init__(
        self,
        emb_dim: int,
        num_buckets: int = MAX_DENSE_BUCKETS,
        value_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        if not 1 <= num_buckets <= MAX_DENSE_BUCKETS:
            raise ValueError(f"num_buckets must be in [1, {MAX_DENSE_BUCKETS}]")
        self.num_buckets = num_buckets
        self.lo, self.hi = value_range
        self.emb = nn.Embedding(num_buckets, emb_dim)

    def forward(self, raw_values: Any) -> torch.Tensor:
        x = torch.as_tensor(
            raw_values, dtype=torch.float32, device=self.emb.weight.device
        ).reshape(-1)
        x01 = ((x - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)  # normalize
        idx = (x01 * (self.num_buckets - 1)).floor().long()          # bucketize
        return self.emb(idx)  # [B, emb_dim]


class GeoEmbedding(nn.Module):
    """(lat, lng) -> latitude and longitude bucketized/embedded separately."""

    def __init__(self, emb_dim: int, num_buckets: int = MAX_DENSE_BUCKETS) -> None:
        super().__init__()
        self.lat = DenseBucketEmbedding(emb_dim, num_buckets, (-90.0, 90.0))
        self.lng = DenseBucketEmbedding(emb_dim, num_buckets, (-180.0, 180.0))

    def forward(self, locations: Any) -> torch.Tensor:
        loc = torch.as_tensor(locations, dtype=torch.float32)  # [B, 2]
        return torch.cat([self.lat(loc[:, 0]), self.lng(loc[:, 1])], dim=-1)


class PretrainedProjection(nn.Module):
    """Embedding from a known upstream model, projected to the shared width."""

    def __init__(self, in_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, emb_dim)

    def forward(self, pretrained_emb: Any) -> torch.Tensor:
        x = torch.as_tensor(pretrained_emb, dtype=torch.float32)
        return self.proj(x)  # [B, emb_dim]


class WindowPatternEmbedding(nn.Module):
    """(1d, 1w, 1m) binary flags -> 3-bit pattern [0..7] -> embedding."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(8, emb_dim)

    def forward(self, windows: Any) -> torch.Tensor:
        w = torch.as_tensor(windows, device=self.emb.weight.device).long()  # [B, 3]
        pattern = w[:, 0] * 4 + w[:, 1] * 2 + w[:, 2]
        return self.emb(pattern)  # [B, emb_dim]


class PooledHashEmbedding(nn.Module):
    """Variable-length id list -> hash each id -> embedding pooling.

    Classic recipe for list features (clicked video ids, past-week clicked
    user ids): per-id hash embedding, then mean/sum pooling to a fixed
    [B, emb_dim] so it can join the concat like any other feature.
    Empty lists pool to a zero vector.
    """

    def __init__(self, num_buckets: int, emb_dim: int, mode: str = "mean") -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.EmbeddingBag(num_buckets, emb_dim, mode=mode)

    def forward(self, id_lists: Sequence[Sequence[Any]]) -> torch.Tensor:
        device = self.emb.weight.device
        flat, offsets = [], []
        for ids in id_lists:
            offsets.append(len(flat))
            flat.extend(stable_hash_bucket(v, self.num_buckets) for v in ids)
        flat_t = torch.tensor(flat, dtype=torch.long, device=device)
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)
        return self.emb(flat_t, offsets_t)  # [B, emb_dim]


class UserTowerEncoder(nn.Module):
    """UserInfo -> [B, output_dim].

    sparse (hash -> single embedding layer) + dense (normalize -> bucket)
    all concatenated, then MLP down to output_dim.
    """

    def __init__(
        self,
        emb_dim: int = 16,
        output_dim: int = 128,
        hidden_dim: int = 256,
        user_hash_buckets: int = 10000,
        pretrained_dim: int = 32,
    ) -> None:
        super().__init__()
        self.user_id_emb = HashEmbedding(user_hash_buckets, emb_dim)
        self.age_emb = DenseBucketEmbedding(emb_dim, 100, (0.0, 100.0))
        self.geo_emb = GeoEmbedding(emb_dim)
        self.pretrained_proj = PretrainedProjection(pretrained_dim, emb_dim)
        self.pretrained_dim = pretrained_dim

        concat_dim = emb_dim * 5  # id + age + geo(2x) + pretrained
        # LayerNorm on the concat and the hidden layer keeps activation
        # scales stable regardless of each feature embedding's magnitude
        self.mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, user_info: UserInfo) -> torch.Tensor:
        id_emb = self.user_id_emb(user_info.user_id)
        batch = id_emb.size(0)
        if user_info.user_embedding is not None:
            pre = self.pretrained_proj(user_info.user_embedding)
        else:
            pre = id_emb.new_zeros(batch, self.pretrained_proj.proj.out_features)
        parts = [
            id_emb,
            self.age_emb(user_info.user_age),
            self.geo_emb(user_info.user_location),
            pre,
        ]
        return self.mlp(torch.cat(parts, dim=-1))  # [B, output_dim]


class VideoTowerEncoder(nn.Module):
    """VideoInfo -> [B, output_dim]. Same recipe as the user tower."""

    def __init__(
        self,
        emb_dim: int = 16,
        output_dim: int = 128,
        hidden_dim: int = 256,
        video_hash_buckets: int = 10000,
        sparse_hash_buckets: int = 1000,
        title_dim: int = 32,
        tags_dim: int = 32,
        max_watch_time: float = 7200.0,
    ) -> None:
        super().__init__()
        self.video_id_emb = HashEmbedding(video_hash_buckets, emb_dim)
        self.creator_emb = HashEmbedding(sparse_hash_buckets, emb_dim)
        self.comment_count_emb = HashEmbedding(sparse_hash_buckets, emb_dim)
        self.like_count_emb = HashEmbedding(sparse_hash_buckets, emb_dim)
        self.title_proj = PretrainedProjection(title_dim, emb_dim)
        self.tags_proj = PretrainedProjection(tags_dim, emb_dim)
        self.sentiment_emb = DenseBucketEmbedding(emb_dim, 100, (0.0, 1.0))
        self.watch_time_emb = DenseBucketEmbedding(emb_dim, 100, (0.0, max_watch_time))

        concat_dim = emb_dim * 8
        self.mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, video_info: VideoInfo) -> torch.Tensor:
        parts = [
            self.video_id_emb(video_info.video_id),
            self.creator_emb(video_info.creator_id),
            self.comment_count_emb(video_info.comment_count),
            self.like_count_emb(video_info.like_count),
            self.title_proj(video_info.video_title_embedding),
            self.tags_proj(video_info.video_tags_embedding),
            self.sentiment_emb(video_info.comment_sentiment),
            self.watch_time_emb(video_info.avg_watch_time),
        ]
        return self.mlp(torch.cat(parts, dim=-1))  # [B, output_dim]


class InteractionEncoder(nn.Module):
    """UserVideoInteractiveInfo -> [B, 7 * emb_dim] concat encoding.

    user_id / video_id hashed separately; each event type (click / like /
    comment / dislike) gets its own window-pattern embedding input;
    watch_time dense-bucketized. ``scenario`` is ignored for now.

    TODO(big-layer-2): what sits on top of this concat is undecided.
    """

    def __init__(
        self,
        emb_dim: int = 16,
        id_hash_buckets: int = 10000,
        max_watch_time: float = 7200.0,
    ) -> None:
        super().__init__()
        self.user_id_emb = HashEmbedding(id_hash_buckets, emb_dim)
        self.video_id_emb = HashEmbedding(id_hash_buckets, emb_dim)
        self.clicked_emb = WindowPatternEmbedding(emb_dim)
        self.liked_emb = WindowPatternEmbedding(emb_dim)
        self.commented_emb = WindowPatternEmbedding(emb_dim)
        self.disliked_emb = WindowPatternEmbedding(emb_dim)
        self.watch_time_emb = DenseBucketEmbedding(emb_dim, 100, (0.0, max_watch_time))
        self.output_dim = emb_dim * 7
        self.norm = nn.LayerNorm(self.output_dim)  # stabilize the concat output

    def forward(self, interactive_info: UserVideoInteractiveInfo) -> torch.Tensor:
        parts = [
            self.user_id_emb(interactive_info.user_id),
            self.video_id_emb(interactive_info.video_id),
            self.clicked_emb(interactive_info.clicked_windows),
            self.liked_emb(interactive_info.liked_windows),
            self.commented_emb(interactive_info.commented_windows),
            self.disliked_emb(interactive_info.disliked_windows),
            self.watch_time_emb(interactive_info.watch_time),
        ]
        return self.norm(torch.cat(parts, dim=-1))  # [B, 7 * emb_dim]
