from .features import (
    ExtensibleInfo,
    UserInfo,
    UserVideoInteractiveInfo,
    VideoInfo,
)

__all__ = [
    "ExtensibleInfo",
    "UserInfo",
    "VideoInfo",
    "UserVideoInteractiveInfo",
]

# torch-dependent exports are optional so feature-only usage works without torch
try:
    from .loss import in_batch_negative_ce_loss  # noqa: F401
    from .model import TTSNConfig, TwoTowerModel  # noqa: F401

    __all__ += ["TTSNConfig", "TwoTowerModel", "in_batch_negative_ce_loss"]
except ImportError:
    pass
