"""Feature encoding config for the TTSN model.

Conventions:
- Feature name: ``feature_<ClassName>_<field_name>``, e.g. ``feature_UserInfo_user_age``.
- Feature id: integers from 10000 up, one 100-wide block per info class so each
  class can grow without renumbering:

    UserInfo                  10000 - 10099
    VideoInfo                 10100 - 10199
    UserVideoInteractiveInfo  10200 - 10299
    (experimental/extensions) 19000 - 19999

  Retired ids are NEVER reused (protobuf rule).
- ``dtype`` is the Python type of the raw feature value (before tensorization).
- ``encoding`` describes how the raw value becomes model input:
    hash:N          sparse -> stable hash into N buckets -> embedding
    dense_bucket:N  dense -> normalize to [0,1] -> bucketize (N <= 100) -> embedding
    geo             (lat, lng) -> latitude/longitude bucketized separately
    pretrained      embedding produced by an upstream/known model
    windows         [1d, 1w, 1m] binary pattern -> embedding
    none            carried but not encoded (e.g. display-only)

TODO(feature-capture): wire upstream feature ingestion. For now we assume all
feature values are already materialized on the info objects. Later:
  - feature store client: fetch by (feature_id, entity_id) with TTL/caching
  - user profile service: hydrate UserInfo (age/location/long-term embedding)
  - realtime signals: interaction windows / watch_time from serving context
  - backfill + point-in-time-correct joins for offline training data

TODO(features, deferred):
  - UserInfo: user group/region, preference categories, clicked/commented
    video id lists
  - VideoInfo: past-week clicked user ids (hashed list), watch-time
    demographic distribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

from .features import UserInfo, UserVideoInteractiveInfo, VideoInfo

# id block starts
USER_INFO_BLOCK = 10000
VIDEO_INFO_BLOCK = 10100
INTERACTIVE_INFO_BLOCK = 10200
EXPERIMENTAL_BLOCK = 19000
BLOCK_SIZE = 100

# ids that were used in the past and must never be reassigned
RETIRED_FEATURE_IDS = {
    10204,  # feature_UserVideoInteractiveInfo_historical_clicks (removed)
    10205,  # feature_UserVideoInteractiveInfo_comment_info (moved to VideoInfo)
}


@dataclass(frozen=True, kw_only=True)
class FeatureSpec:
    feature_id: int
    name: str                  # feature_<ClassName>_<field_name>
    dtype: type                # python type of the raw value
    owner: Type[Any]           # info class this feature belongs to
    field_name: str            # attribute name on the owner class
    encoding: str = "none"     # how the raw value becomes model input
    description: str = ""


def make_feature_name(owner: Type[Any], field_name: str) -> str:
    return f"feature_{owner.__name__}_{field_name}"


class FeatureRegistry:
    """Keeps all FeatureSpecs, enforcing unique ids and names."""

    def __init__(self) -> None:
        self._by_id: Dict[int, FeatureSpec] = {}
        self._by_name: Dict[str, FeatureSpec] = {}

    def register(
        self,
        *,
        feature_id: int,
        owner: Type[Any],
        field_name: str,
        dtype: type,
        encoding: str = "none",
        description: str = "",
    ) -> FeatureSpec:
        name = make_feature_name(owner, field_name)
        if feature_id in RETIRED_FEATURE_IDS:
            raise ValueError(f"feature_id {feature_id} is retired, never reuse it")
        if feature_id in self._by_id:
            raise ValueError(
                f"duplicate feature_id {feature_id}: already used by "
                f"{self._by_id[feature_id].name}"
            )
        if name in self._by_name:
            raise ValueError(f"duplicate feature name {name}")
        spec = FeatureSpec(
            feature_id=feature_id,
            name=name,
            dtype=dtype,
            owner=owner,
            field_name=field_name,
            encoding=encoding,
            description=description,
        )
        self._by_id[feature_id] = spec
        self._by_name[name] = spec
        return spec

    def by_id(self, feature_id: int) -> FeatureSpec:
        return self._by_id[feature_id]

    def by_name(self, name: str) -> FeatureSpec:
        return self._by_name[name]

    def specs_for(self, owner: Type[Any]) -> List[FeatureSpec]:
        return [s for s in self._by_name.values() if s.owner is owner]

    def all_specs(self) -> List[FeatureSpec]:
        return sorted(self._by_name.values(), key=lambda s: s.feature_id)

    def __len__(self) -> int:
        return len(self._by_id)


# ---------------------------------------------------------------------------
# Global registry with the current production schema.
# ---------------------------------------------------------------------------
REGISTRY = FeatureRegistry()

# UserInfo block: 10000-10099
FEATURE_USER_ID = REGISTRY.register(
    feature_id=10001, owner=UserInfo, field_name="user_id",
    dtype=str, encoding="hash:10000", description="unique user id")
FEATURE_USER_EMBEDDING = REGISTRY.register(
    feature_id=10002, owner=UserInfo, field_name="user_embedding",
    dtype=list, encoding="pretrained", description="pretrained user embedding")
FEATURE_USER_AGE = REGISTRY.register(
    feature_id=10003, owner=UserInfo, field_name="user_age",
    dtype=int, encoding="dense_bucket:100", description="age, normalized then bucketized")
FEATURE_USER_NAME = REGISTRY.register(
    feature_id=10004, owner=UserInfo, field_name="user_name",
    dtype=str, encoding="none", description="display name, not encoded")
FEATURE_USER_LOCATION = REGISTRY.register(
    feature_id=10005, owner=UserInfo, field_name="user_location",
    dtype=tuple, encoding="geo",
    description="(lat, lng); latitude/longitude bucketized separately")

# VideoInfo block: 10100-10199
FEATURE_VIDEO_ID = REGISTRY.register(
    feature_id=10101, owner=VideoInfo, field_name="video_id",
    dtype=str, encoding="hash:10000", description="unique video id")
FEATURE_VIDEO_EMBEDDING = REGISTRY.register(
    feature_id=10102, owner=VideoInfo, field_name="video_embedding",
    dtype=list, encoding="pretrained", description="content embedding")
FEATURE_VIDEO_AUDIO_EMBEDDING = REGISTRY.register(
    feature_id=10103, owner=VideoInfo, field_name="video_audio_embedding",
    dtype=list, encoding="pretrained", description="audio-track embedding")
FEATURE_VIDEO_TITLE = REGISTRY.register(
    feature_id=10104, owner=VideoInfo, field_name="video_title",
    dtype=str, encoding="none", description="raw title text, not encoded directly")
FEATURE_VIDEO_TAGS = REGISTRY.register(
    feature_id=10105, owner=VideoInfo, field_name="video_tags",
    dtype=list, encoding="none", description="raw tags, not encoded directly")
FEATURE_VIDEO_CREATOR_ID = REGISTRY.register(
    feature_id=10106, owner=VideoInfo, field_name="creator_id",
    dtype=str, encoding="hash:1000", description="creator/channel id")
FEATURE_VIDEO_TITLE_EMBEDDING = REGISTRY.register(
    feature_id=10107, owner=VideoInfo, field_name="video_title_embedding",
    dtype=list, encoding="pretrained", description="title embedding from known text model")
FEATURE_VIDEO_TAGS_EMBEDDING = REGISTRY.register(
    feature_id=10108, owner=VideoInfo, field_name="video_tags_embedding",
    dtype=list, encoding="pretrained", description="tags embedding from known model")
FEATURE_VIDEO_COMMENT_COUNT = REGISTRY.register(
    feature_id=10109, owner=VideoInfo, field_name="comment_count",
    dtype=int, encoding="hash:1000", description="number of comments, sparse")
FEATURE_VIDEO_LIKE_COUNT = REGISTRY.register(
    feature_id=10110, owner=VideoInfo, field_name="like_count",
    dtype=int, encoding="hash:1000", description="number of likes, sparse")
FEATURE_VIDEO_COMMENT_SENTIMENT = REGISTRY.register(
    feature_id=10111, owner=VideoInfo, field_name="comment_sentiment",
    dtype=float, encoding="dense_bucket:100",
    description="aggregated comment polarity in [0,1]; raw text not used")
FEATURE_VIDEO_AVG_WATCH_TIME = REGISTRY.register(
    feature_id=10112, owner=VideoInfo, field_name="avg_watch_time",
    dtype=float, encoding="dense_bucket:100", description="average watch time, seconds")

# UserVideoInteractiveInfo block: 10200-10299 (10204/10205 retired)
FEATURE_INTERACTIVE_USER_ID = REGISTRY.register(
    feature_id=10201, owner=UserVideoInteractiveInfo, field_name="user_id",
    dtype=str, encoding="hash:10000", description="user id of the interaction")
FEATURE_INTERACTIVE_VIDEO_ID = REGISTRY.register(
    feature_id=10202, owner=UserVideoInteractiveInfo, field_name="video_id",
    dtype=str, encoding="hash:10000", description="video id of the interaction")
FEATURE_INTERACTIVE_SCENARIO = REGISTRY.register(
    feature_id=10203, owner=UserVideoInteractiveInfo, field_name="scenario",
    dtype=str, encoding="none",
    description="where the click happened; ignored in encoding for now")
FEATURE_INTERACTIVE_CLICKED_WINDOWS = REGISTRY.register(
    feature_id=10206, owner=UserVideoInteractiveInfo, field_name="clicked_windows",
    dtype=list, encoding="windows", description="clicked this video in 1d/1w/1m")
FEATURE_INTERACTIVE_LIKED_WINDOWS = REGISTRY.register(
    feature_id=10207, owner=UserVideoInteractiveInfo, field_name="liked_windows",
    dtype=list, encoding="windows", description="liked this video in 1d/1w/1m")
FEATURE_INTERACTIVE_COMMENTED_WINDOWS = REGISTRY.register(
    feature_id=10208, owner=UserVideoInteractiveInfo, field_name="commented_windows",
    dtype=list, encoding="windows", description="commented this video in 1d/1w/1m")
FEATURE_INTERACTIVE_DISLIKED_WINDOWS = REGISTRY.register(
    feature_id=10209, owner=UserVideoInteractiveInfo, field_name="disliked_windows",
    dtype=list, encoding="windows", description="disliked this video in 1d/1w/1m")
FEATURE_INTERACTIVE_WATCH_TIME = REGISTRY.register(
    feature_id=10210, owner=UserVideoInteractiveInfo, field_name="watch_time",
    dtype=float, encoding="dense_bucket:100", description="watch time of this pair, seconds")


def encode_info(info: Any) -> Dict[int, Any]:
    """Encode an info object into {feature_id: raw_value}.

    Only registered, non-None schema fields are emitted; extensions are skipped
    until they are promoted into the registry (experimental block 19000+).
    """
    out: Dict[int, Any] = {}
    for spec in REGISTRY.specs_for(type(info)):
        value = getattr(info, spec.field_name, None)
        if value is not None:
            out[spec.feature_id] = value
    return out
