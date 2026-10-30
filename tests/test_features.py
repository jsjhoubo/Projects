import unittest
from dataclasses import dataclass
from typing import Any

from ttsn.features import UserInfo, UserVideoInteractiveInfo, VideoInfo


class TestUserInfo(unittest.TestCase):
    def test_basic_construction_and_defaults(self):
        u = UserInfo(user_id="u1", user_age=30, user_location="SF")
        self.assertEqual(u.user_id, "u1")
        self.assertEqual(u.user_age, 30)
        self.assertEqual(u.user_location, "SF")
        self.assertIsNone(u.user_embedding)
        self.assertIsNone(u.user_name)
        self.assertEqual(u.extensions, {})

    def test_extensions_roundtrip(self):
        u = UserInfo(user_id="u1")
        self.assertFalse(u.has_extension("device_type"))
        self.assertEqual(u.get_extension("device_type", "unknown"), "unknown")

        u.set_extension("device_type", "roku")
        u.set_extension("watch_minutes_7d", 123.5)
        self.assertTrue(u.has_extension("device_type"))
        self.assertEqual(u.get_extension("device_type"), "roku")
        self.assertEqual(u.get_extension("watch_minutes_7d"), 123.5)

    def test_subclass_extension(self):
        # schema evolution via inheritance, protobuf-extend style
        @dataclass(kw_only=True)
        class VipUserInfo(UserInfo):
            membership_level: int = 0
            payment_history: Any = None

        v = VipUserInfo(user_id="u2", user_age=25, membership_level=3)
        self.assertIsInstance(v, UserInfo)
        self.assertEqual(v.membership_level, 3)
        self.assertEqual(v.user_age, 25)
        # extensions map still works on subclasses
        v.set_extension("ab_bucket", "treatment")
        self.assertEqual(v.get_extension("ab_bucket"), "treatment")

    def test_to_dict_merges_fields_and_extensions(self):
        u = UserInfo(user_id="u1", user_age=30)
        u.set_extension("device_type", "roku")
        u.set_extension("user_age", -1)  # collision: schema field must win

        d = u.to_dict()
        self.assertEqual(d["user_id"], "u1")
        self.assertEqual(d["user_age"], 30)
        self.assertEqual(d["device_type"], "roku")
        self.assertNotIn("extensions", d)


class TestVideoInfo(unittest.TestCase):
    def test_basic_construction(self):
        v = VideoInfo(
            video_id="v9",
            video_embedding=[0.1, 0.2],
            video_audio_embedding=[0.3],
            video_tags=["comedy", "2024"],
        )
        self.assertEqual(v.video_id, "v9")
        self.assertEqual(v.video_embedding, [0.1, 0.2])
        self.assertEqual(v.video_audio_embedding, [0.3])
        self.assertEqual(v.video_tags, ["comedy", "2024"])
        self.assertIsNone(v.video_title)

    def test_subclass_extension(self):
        @dataclass(kw_only=True)
        class LiveVideoInfo(VideoInfo):
            is_live: bool = True
            concurrent_viewers: int = 0

        lv = LiveVideoInfo(video_id="live1", concurrent_viewers=4200)
        self.assertIsInstance(lv, VideoInfo)
        self.assertTrue(lv.is_live)
        self.assertEqual(lv.concurrent_viewers, 4200)


class TestUserVideoInteractiveInfo(unittest.TestCase):
    def test_basic_construction(self):
        it = UserVideoInteractiveInfo(
            user_id="u1",
            video_id="v9",
            scenario="homepage",
            clicked_windows=[1, 1, 1],
            liked_windows=[0, 0, 1],
            commented_windows=[0, 0, 0],
            disliked_windows=[0, 0, 0],
            watch_time=432.0,
        )
        self.assertEqual(it.user_id, "u1")
        self.assertEqual(it.video_id, "v9")
        self.assertEqual(it.scenario, "homepage")
        self.assertEqual(it.clicked_windows, [1, 1, 1])
        self.assertEqual(it.liked_windows, [0, 0, 1])
        self.assertEqual(it.watch_time, 432.0)

    def test_extension_for_new_cross_feature(self):
        it = UserVideoInteractiveInfo(user_id="u1", video_id="v9")
        it.set_extension("co_watch_score", 0.87)
        self.assertAlmostEqual(it.get_extension("co_watch_score"), 0.87)


if __name__ == "__main__":
    unittest.main()
