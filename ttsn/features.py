"""Feature containers for the two-tower (TTSN) model.

Design notes:
- All info classes are keyword-only dataclasses, so subclasses can freely add
  new fields (schema evolution by inheritance).
- Besides inheritance, each class carries a protobuf-style ``extensions`` map
  for ad-hoc / experimental features that don't warrant a schema change yet.
- Tensor-valued fields (embeddings, histories) are typed as ``Any`` on purpose:
  they may hold ``torch.Tensor``, ``numpy.ndarray`` or plain lists, and this
  module stays importable without torch installed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(kw_only=True)
class ExtensibleInfo:
    """Base for all feature containers.

    ``extensions`` mimics protobuf extensions: a typed-schema escape hatch for
    new features before they are promoted to first-class fields (either here
    or in a subclass).
    """

    extensions: Dict[str, Any] = field(default_factory=dict)

    def set_extension(self, key: str, value: Any) -> None:
        self.extensions[key] = value

    def get_extension(self, key: str, default: Any = None) -> Any:
        return self.extensions.get(key, default)

    def has_extension(self, key: str) -> bool:
        return key in self.extensions

    def to_dict(self) -> Dict[str, Any]:
        """Flatten declared fields + extensions into one dict.

        Declared (schema) fields win over extensions on key collision.
        """
        out: Dict[str, Any] = dict(self.extensions)
        for f in dataclasses.fields(self):
            if f.name == "extensions":
                continue
            out[f.name] = getattr(self, f.name)
        return out


@dataclass(kw_only=True)
class UserInfo(ExtensibleInfo):
    """User-side tower input. Fields may be per-sample or batched tensors."""

    user_id: Any                       # sparse, hash 10000
    user_embedding: Any = None         # pretrained / lookup embedding
    user_age: Optional[Any] = None     # dense -> normalize + bucket
    user_name: Optional[Any] = None    # not encoded
    user_location: Optional[Any] = None  # (lat, lng) pair -> geo encoding

    # TODO(features): user group/region, preference categories,
    # clicked/commented video id lists.


@dataclass(kw_only=True)
class VideoInfo(ExtensibleInfo):
    """Video/item-side tower input.

    Comment info lives here (it's a property of the video): counts and an
    aggregated sentiment polarity — raw comment text is NOT a feature.
    """

    video_id: Any
    video_embedding: Any = None        # content embedding
    video_audio_embedding: Any = None  # audio-track embedding
    video_title: Optional[Any] = None          # raw text (not encoded directly)
    video_title_embedding: Any = None          # from a pretrained text model
    video_tags: Optional[Any] = None           # raw tags (not encoded directly)
    video_tags_embedding: Any = None           # from a pretrained model
    creator_id: Optional[Any] = None           # sparse, hash 1000
    comment_count: Optional[Any] = None        # sparse, hash 1000
    like_count: Optional[Any] = None           # sparse, hash 1000
    comment_sentiment: Optional[Any] = None    # aggregated polarity in [0,1]
    avg_watch_time: Optional[Any] = None       # dense, seconds

    # TODO(features): past-week clicked user ids (hashed list);
    # watch-time demographic distribution.


@dataclass(kw_only=True)
class UserVideoInteractiveInfo(ExtensibleInfo):
    """Cross features for the (user, video) pair.

    Window flags are [B, 3] binaries over (1 day, 1 week, 1 month):
    did this user click/like/comment/dislike this video in that window.
    """

    user_id: Any
    video_id: Any
    scenario: Optional[Any] = None       # where the click happened; ignored in encoding for now
    clicked_windows: Optional[Any] = None     # [B, 3] binary (1d, 1w, 1m)
    liked_windows: Optional[Any] = None       # [B, 3]
    commented_windows: Optional[Any] = None   # [B, 3]
    disliked_windows: Optional[Any] = None    # [B, 3]
    watch_time: Optional[Any] = None          # dense, seconds
