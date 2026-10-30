"""Tests for ttsn/encoding.py — big layer 1 (feature encoding).

What is covered:
- stable_hash_bucket / hash_to_buckets: deterministic hashing, bucket range,
  tensor and python-list inputs
- HashEmbedding: output shape; same raw id -> same embedding row
- DenseBucketEmbedding: normalize-then-bucketize; out-of-range values clamp
  to the edge buckets; values in the same bucket share an embedding;
  bucket count capped at 100
- GeoEmbedding: [B, 2*dim] output; latitude and longitude encoded separately
- WindowPatternEmbedding: (1d,1w,1m) flags -> 3-bit pattern; same pattern
  -> same embedding, different pattern -> different lookup
- PooledHashEmbedding: variable-length id lists pooled to fixed [B, dim];
  empty list -> zero vector; order invariance of mean pooling
- UserTowerEncoder / VideoTowerEncoder: [B, output_dim] output;
  user_embedding=None takes the zero-fill path
- InteractionEncoder: [B, 7*emb_dim] concat output
"""

import unittest

import torch

from ttsn.encoding import (
    DenseBucketEmbedding,
    GeoEmbedding,
    HashEmbedding,
    InteractionEncoder,
    PooledHashEmbedding,
    UserTowerEncoder,
    VideoTowerEncoder,
    WindowPatternEmbedding,
    hash_to_buckets,
    stable_hash_bucket,
)
from ttsn.features import UserInfo, UserVideoInteractiveInfo, VideoInfo

B = 4


def make_user(with_pretrained=True):
    return UserInfo(
        user_id=["u1", "u2", "u3", "u1"],
        user_embedding=torch.randn(B, 32) if with_pretrained else None,
        user_age=[23, 45, 31, 23],
        user_location=[(37.7, -122.4), (40.7, -74.0), (34.0, -118.2), (37.7, -122.4)],
    )


def make_video():
    return VideoInfo(
        video_id=["v1", "v2", "v3", "v4"],
        creator_id=["c1", "c2", "c1", "c3"],
        comment_count=[12, 0, 340, 5],
        like_count=[100, 3, 2000, 40],
        video_title_embedding=torch.randn(B, 32),
        video_tags_embedding=torch.randn(B, 32),
        comment_sentiment=[0.8, 0.4, 0.95, 0.5],
        avg_watch_time=[600.0, 120.0, 1800.0, 300.0],
    )


def make_interaction():
    return UserVideoInteractiveInfo(
        user_id=["u1", "u2", "u3", "u1"],
        video_id=["v1", "v2", "v3", "v4"],
        clicked_windows=[[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
        liked_windows=[[0, 0, 1]] * B,
        commented_windows=[[0, 0, 0]] * B,
        disliked_windows=[[0, 0, 0]] * B,
        watch_time=[432.0, 88.0, 1500.0, 0.0],
    )


class TestStableHash(unittest.TestCase):
    def test_deterministic_and_in_range(self):
        """Same value always hashes to the same bucket, within [0, N)."""
        for v in ["u1", 42, ("a", 1)]:
            b1 = stable_hash_bucket(v, 1000)
            b2 = stable_hash_bucket(v, 1000)
            self.assertEqual(b1, b2)
            self.assertTrue(0 <= b1 < 1000)

    def test_list_and_tensor_inputs(self):
        """hash_to_buckets accepts python lists and int tensors, returns [B] long."""
        from_list = hash_to_buckets(["a", "b", "a"], 100)
        self.assertEqual(from_list.shape, (3,))
        self.assertEqual(from_list[0].item(), from_list[2].item())

        from_tensor = hash_to_buckets(torch.tensor([7, 107, 7]), 100)
        self.assertEqual(from_tensor.dtype, torch.long)
        self.assertEqual(from_tensor[0].item(), from_tensor[2].item())  # 7 == 107 % 100


class TestHashEmbedding(unittest.TestCase):
    def test_shape_and_same_id_same_embedding(self):
        """Output is [B, dim]; identical raw ids share an embedding row."""
        emb = HashEmbedding(1000, 16)
        out = emb(["x", "y", "x"])
        self.assertEqual(out.shape, (3, 16))
        torch.testing.assert_close(out[0], out[2])


class TestDenseBucketEmbedding(unittest.TestCase):
    def test_shape_and_bucket_sharing(self):
        """Values falling in the same bucket get the same embedding."""
        emb = DenseBucketEmbedding(16, num_buckets=10, value_range=(0.0, 100.0))
        out = emb([5.0, 5.5, 95.0])  # 5.0 and 5.5 land in the same bucket
        self.assertEqual(out.shape, (3, 16))
        torch.testing.assert_close(out[0], out[1])
        self.assertFalse(torch.allclose(out[0], out[2]))

    def test_out_of_range_clamps(self):
        """Values outside [lo, hi] clamp to the first/last bucket."""
        emb = DenseBucketEmbedding(8, num_buckets=10, value_range=(0.0, 1.0))
        out = emb([-5.0, 0.0, 1.0, 99.0])
        torch.testing.assert_close(out[0], out[1])  # -5 clamps to bucket of 0
        torch.testing.assert_close(out[2], out[3])  # 99 clamps to bucket of 1

    def test_bucket_cap_100(self):
        """At most 100 buckets are allowed."""
        with self.assertRaises(ValueError):
            DenseBucketEmbedding(8, num_buckets=101)


class TestGeoEmbedding(unittest.TestCase):
    def test_lat_lng_encoded_separately(self):
        """Output is [B, 2*dim]; same lat different lng -> first half equal."""
        emb = GeoEmbedding(16)
        out = emb([(37.7, -122.4), (37.7, 139.7)])
        self.assertEqual(out.shape, (2, 32))
        torch.testing.assert_close(out[0, :16], out[1, :16])          # lat part
        self.assertFalse(torch.allclose(out[0, 16:], out[1, 16:]))    # lng part


class TestWindowPatternEmbedding(unittest.TestCase):
    def test_pattern_lookup(self):
        """(1d,1w,1m) flags form a 3-bit pattern; equal patterns share the
        embedding, different patterns use different table rows."""
        emb = WindowPatternEmbedding(16)
        out = emb([[1, 0, 1], [1, 0, 1], [0, 0, 0]])
        self.assertEqual(out.shape, (3, 16))
        torch.testing.assert_close(out[0], out[1])
        self.assertFalse(torch.allclose(out[0], out[2]))


class TestPooledHashEmbedding(unittest.TestCase):
    def test_variable_lengths_and_empty(self):
        """Rows of different lengths pool to fixed [B, dim]; empty -> zeros."""
        emb = PooledHashEmbedding(1000, 16)
        out = emb([["v1", "v2", "v3"], ["v9"], [], ["v1"] * 5])
        self.assertEqual(out.shape, (4, 16))
        self.assertTrue((out[2] == 0).all())

    def test_mean_pooling_order_invariant(self):
        """Mean pooling ignores the order of ids in the list."""
        emb = PooledHashEmbedding(1000, 16)
        out = emb([["a", "b", "c"], ["c", "a", "b"]])
        torch.testing.assert_close(out[0], out[1])


class TestTowerEncoders(unittest.TestCase):
    def test_user_tower_output_shape(self):
        """UserTowerEncoder maps UserInfo to [B, output_dim]."""
        enc = UserTowerEncoder(output_dim=128)
        self.assertEqual(enc(make_user()).shape, (B, 128))

    def test_user_tower_without_pretrained_embedding(self):
        """user_embedding=None takes the zero-fill path and still encodes."""
        enc = UserTowerEncoder(output_dim=128)
        self.assertEqual(enc(make_user(with_pretrained=False)).shape, (B, 128))

    def test_video_tower_output_shape(self):
        """VideoTowerEncoder maps VideoInfo to [B, output_dim]."""
        enc = VideoTowerEncoder(output_dim=128)
        self.assertEqual(enc(make_video()).shape, (B, 128))

    def test_interaction_encoder_concat_width(self):
        """InteractionEncoder outputs the concat of 7 blocks: user_id,
        video_id, 4 event window patterns, watch_time -> [B, 7*emb_dim]."""
        enc = InteractionEncoder(emb_dim=16)
        out = enc(make_interaction())
        self.assertEqual(out.shape, (B, 7 * 16))
        self.assertEqual(enc.output_dim, 7 * 16)


if __name__ == "__main__":
    unittest.main()
