"""Mock interaction-log data for TTSN training.

Layout mirrors the real pipeline:
- entity tables: per-user and per-video attribute rows (what a feature store
  / profile service would serve)
- interaction log: 30 days of (user, video, day, watch_time, window flags)
  click events
- temporal split: last N days are the test set (never split randomly —
  that leaks future information into training)
- batching: each batch materializes UserInfo / VideoInfo /
  UserVideoInteractiveInfo plus per-sample weights for one gradient step

The log has latent structure (user/video affinity vectors drive which pairs
click and how long they watch) so the model has something real to learn;
pretrained embeddings are noisy views of the latents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch

from .features import UserInfo, UserVideoInteractiveInfo, VideoInfo


@dataclass(kw_only=True)
class MockDataConfig:
    num_users: int = 500
    num_videos: int = 300
    days: int = 30
    interactions_per_day: int = 1000
    test_days: int = 2            # last days held out for testing
    latent_dim: int = 8
    pretrained_dim: int = 32
    seed: int = 0


def generate_mock_data(cfg: MockDataConfig) -> Dict[str, torch.Tensor]:
    """Entity tables + 30-day interaction log, all as tensors keyed by name."""
    g = torch.Generator().manual_seed(cfg.seed)

    # latent affinity space shared by users and videos
    user_latent = torch.randn(cfg.num_users, cfg.latent_dim, generator=g)
    video_latent = torch.randn(cfg.num_videos, cfg.latent_dim, generator=g)

    # ---- user table ----
    user_age = torch.randint(13, 80, (cfg.num_users,), generator=g).float()
    user_lat = torch.rand(cfg.num_users, generator=g) * 180 - 90
    user_lng = torch.rand(cfg.num_users, generator=g) * 360 - 180
    user_pretrained = torch.cat(
        [user_latent, torch.randn(cfg.num_users, cfg.pretrained_dim - cfg.latent_dim, generator=g) * 0.1],
        dim=1,
    )

    # ---- video table ----
    video_creator = torch.randint(0, 200, (cfg.num_videos,), generator=g)
    video_comment_count = torch.randint(0, 500, (cfg.num_videos,), generator=g)
    video_like_count = torch.randint(0, 5000, (cfg.num_videos,), generator=g)
    video_sentiment = torch.rand(cfg.num_videos, generator=g)
    video_avg_watch = torch.rand(cfg.num_videos, generator=g) * 3600
    video_title_emb = torch.cat(
        [video_latent, torch.randn(cfg.num_videos, cfg.pretrained_dim - cfg.latent_dim, generator=g) * 0.1],
        dim=1,
    )
    video_tags_emb = video_title_emb + torch.randn(cfg.num_videos, cfg.pretrained_dim, generator=g) * 0.1

    # ---- interaction log: users click videos they have affinity with ----
    total = cfg.days * cfg.interactions_per_day
    log_user = torch.randint(0, cfg.num_users, (total,), generator=g)
    # sample 8 candidate videos per event, pick the best-affinity one
    candidates = torch.randint(0, cfg.num_videos, (total, 8), generator=g)
    affinity = (user_latent[log_user].unsqueeze(1) * video_latent[candidates]).sum(-1)
    log_video = candidates.gather(1, affinity.argmax(1, keepdim=True)).squeeze(1)

    log_day = torch.arange(total) // cfg.interactions_per_day
    best_affinity = affinity.max(1).values
    log_watch = (torch.sigmoid(best_affinity) * 3600 + torch.randn(total, generator=g) * 120).clamp(0, 7200)

    # window flags: mildly correlated with affinity (repeat engagers)
    p = torch.sigmoid(best_affinity).unsqueeze(1).expand(total, 3)
    log_clicked = (torch.rand(total, 3, generator=g) < p * 0.8).long()
    log_liked = (torch.rand(total, 3, generator=g) < p * 0.3).long()
    log_commented = (torch.rand(total, 3, generator=g) < p * 0.1).long()
    log_disliked = (torch.rand(total, 3, generator=g) < (1 - p) * 0.05).long()

    # per-sample weight: longer watch -> more trustworthy positive, in [1, 2]
    log_weight = 1.0 + log_watch / 7200.0

    return {
        "user_age": user_age, "user_lat": user_lat, "user_lng": user_lng,
        "user_pretrained": user_pretrained,
        "video_creator": video_creator, "video_comment_count": video_comment_count,
        "video_like_count": video_like_count, "video_sentiment": video_sentiment,
        "video_avg_watch": video_avg_watch, "video_title_emb": video_title_emb,
        "video_tags_emb": video_tags_emb,
        "log_user": log_user, "log_video": log_video, "log_day": log_day,
        "log_watch": log_watch, "log_clicked": log_clicked, "log_liked": log_liked,
        "log_commented": log_commented, "log_disliked": log_disliked,
        "log_weight": log_weight,
    }


def temporal_split(
    data: Dict[str, torch.Tensor], cfg: MockDataConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(train_indices, test_indices): last ``test_days`` days are the test set."""
    cutoff = cfg.days - cfg.test_days
    day = data["log_day"]
    return (day < cutoff).nonzero().squeeze(1), (day >= cutoff).nonzero().squeeze(1)


def make_batch(
    data: Dict[str, torch.Tensor], idx: torch.Tensor
) -> Tuple[UserInfo, VideoInfo, UserVideoInteractiveInfo, torch.Tensor]:
    """Materialize one batch of info objects + sample weights from log rows."""
    u = data["log_user"][idx]
    v = data["log_video"][idx]
    user = UserInfo(
        user_id=u,
        user_embedding=data["user_pretrained"][u],
        user_age=data["user_age"][u],
        user_location=torch.stack([data["user_lat"][u], data["user_lng"][u]], dim=1),
    )
    video = VideoInfo(
        video_id=v,
        creator_id=data["video_creator"][v],
        comment_count=data["video_comment_count"][v],
        like_count=data["video_like_count"][v],
        comment_sentiment=data["video_sentiment"][v],
        avg_watch_time=data["video_avg_watch"][v],
        video_title_embedding=data["video_title_emb"][v],
        video_tags_embedding=data["video_tags_emb"][v],
    )
    inter = UserVideoInteractiveInfo(
        user_id=u,
        video_id=v,
        clicked_windows=data["log_clicked"][idx],
        liked_windows=data["log_liked"][idx],
        commented_windows=data["log_commented"][idx],
        disliked_windows=data["log_disliked"][idx],
        watch_time=data["log_watch"][idx],
    )
    return user, video, inter, data["log_weight"][idx]


def iter_batches(
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    """Yield index slices; one epoch = every batch yielded exactly once."""
    if shuffle:
        perm = torch.randperm(len(indices), generator=generator)
        indices = indices[perm]
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]
