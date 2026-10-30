import unittest

from ttsn.feature_config import (
    EXPERIMENTAL_BLOCK,
    REGISTRY,
    FeatureRegistry,
    encode_info,
    make_feature_name,
)
from ttsn.features import UserInfo, UserVideoInteractiveInfo, VideoInfo


class TestFeatureNaming(unittest.TestCase):
    def test_name_convention(self):
        self.assertEqual(
            make_feature_name(UserInfo, "user_age"), "feature_UserInfo_user_age"
        )
        self.assertEqual(
            make_feature_name(VideoInfo, "video_audio_embedding"),
            "feature_VideoInfo_video_audio_embedding",
        )


class TestRegistry(unittest.TestCase):
    def test_ids_start_at_10000_and_unique(self):
        specs = REGISTRY.all_specs()
        self.assertGreater(len(specs), 0)
        ids = [s.feature_id for s in specs]
        self.assertTrue(all(i >= 10000 for i in ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_lookup_by_id_and_name(self):
        spec = REGISTRY.by_name("feature_UserInfo_user_age")
        self.assertEqual(spec.feature_id, 10003)
        self.assertIs(spec.dtype, int)
        self.assertIs(REGISTRY.by_id(10003), spec)

    def test_specs_for_each_class(self):
        self.assertEqual(len(REGISTRY.specs_for(UserInfo)), 5)
        self.assertEqual(len(REGISTRY.specs_for(VideoInfo)), 12)
        self.assertEqual(len(REGISTRY.specs_for(UserVideoInteractiveInfo)), 8)

    def test_retired_ids_rejected(self):
        r = FeatureRegistry()
        with self.assertRaises(ValueError):
            r.register(feature_id=10204, owner=UserVideoInteractiveInfo,
                       field_name="watch_time", dtype=float)

    def test_encoding_metadata(self):
        self.assertEqual(REGISTRY.by_name("feature_UserInfo_user_id").encoding,
                         "hash:10000")
        self.assertEqual(REGISTRY.by_name("feature_VideoInfo_creator_id").encoding,
                         "hash:1000")
        self.assertEqual(REGISTRY.by_name("feature_UserInfo_user_location").encoding,
                         "geo")
        self.assertEqual(
            REGISTRY.by_name("feature_UserVideoInteractiveInfo_clicked_windows").encoding,
            "windows")

    def test_duplicate_id_rejected(self):
        r = FeatureRegistry()
        r.register(feature_id=EXPERIMENTAL_BLOCK, owner=UserInfo,
                   field_name="user_age", dtype=int)
        with self.assertRaises(ValueError):
            r.register(feature_id=EXPERIMENTAL_BLOCK, owner=UserInfo,
                       field_name="user_name", dtype=str)

    def test_duplicate_name_rejected(self):
        r = FeatureRegistry()
        r.register(feature_id=EXPERIMENTAL_BLOCK, owner=UserInfo,
                   field_name="user_age", dtype=int)
        with self.assertRaises(ValueError):
            r.register(feature_id=EXPERIMENTAL_BLOCK + 1, owner=UserInfo,
                       field_name="user_age", dtype=int)


class TestEncodeInfo(unittest.TestCase):
    def test_encode_user_info_skips_none(self):
        u = UserInfo(user_id="u1", user_age=30)
        encoded = encode_info(u)
        self.assertEqual(encoded[10001], "u1")
        self.assertEqual(encoded[10003], 30)
        self.assertNotIn(10004, encoded)  # user_name is None
        self.assertNotIn(10005, encoded)  # user_location is None

    def test_encode_interactive_info(self):
        it = UserVideoInteractiveInfo(
            user_id="u1", video_id="v9", scenario="homepage",
            clicked_windows=[1, 0, 1], watch_time=120.0,
        )
        encoded = encode_info(it)
        self.assertEqual(encoded[10201], "u1")
        self.assertEqual(encoded[10202], "v9")
        self.assertEqual(encoded[10203], "homepage")
        self.assertEqual(encoded[10206], [1, 0, 1])
        self.assertEqual(encoded[10210], 120.0)
        self.assertNotIn(10207, encoded)  # liked_windows is None
        self.assertNotIn(10204, encoded)  # retired id never emitted


if __name__ == "__main__":
    unittest.main()
