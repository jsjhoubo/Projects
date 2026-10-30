import unittest

import torch

from ttsn.loss import build_positive_mask, in_batch_negative_ce_loss

BATCH = 4


class TestInBatchNegativeLoss(unittest.TestCase):
    def test_scalar_and_positive(self):
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        loss = in_batch_negative_ce_loss(u, v)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0.0)

    def test_perfect_alignment_gives_low_loss(self):
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        aligned = in_batch_negative_ce_loss(u, u, temperature=0.05)
        shuffled = in_batch_negative_ce_loss(u, u.flip(0), temperature=0.05)
        self.assertLess(aligned.item(), shuffled.item())

    def test_uniform_weights_match_unweighted(self):
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        unweighted = in_batch_negative_ce_loss(u, v)
        weighted = in_batch_negative_ce_loss(u, v, sample_weights=torch.full((BATCH,), 3.0))
        torch.testing.assert_close(unweighted, weighted)

    def test_zero_weight_drops_sample(self):
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        w = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss = in_batch_negative_ce_loss(u, v, sample_weights=w)
        logits = u @ v.t() / 0.05
        expected = torch.nn.functional.cross_entropy(
            logits, torch.arange(BATCH), reduction="none"
        )[0]
        torch.testing.assert_close(loss, expected)

    def test_weight_shape_validated(self):
        u = torch.randn(BATCH, 8)
        with self.assertRaises(ValueError):
            in_batch_negative_ce_loss(u, u, sample_weights=torch.ones(BATCH + 1))


class TestTrueLabels(unittest.TestCase):
    def test_build_positive_mask_from_string_user_ids(self):
        mask = build_positive_mask(["u1", "u1", "u2", "u3"])
        expected = torch.tensor([
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_build_positive_mask_from_tensor_user_ids(self):
        mask = build_positive_mask(torch.tensor([7, 9, 7, 3]))
        self.assertTrue(mask[0, 2].item() and mask[2, 0].item())
        self.assertFalse(mask[0, 1].item())
        self.assertTrue(mask.diagonal().all())

    def test_distinct_users_reduce_to_diagonal_ce(self):
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        with_ids = in_batch_negative_ce_loss(u, v, user_ids=["a", "b", "c", "d"])
        diagonal = in_batch_negative_ce_loss(u, v)
        torch.testing.assert_close(with_ids, diagonal)

    def test_same_user_two_docs_elementwise_ce(self):
        # rows 0 and 1 are the same user who clicked d0 and d1:
        # CE_row0 = -(label * log p) summed = -(log p(u,d0) + log p(u,d1))
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        user_ids = ["u1", "u1", "u2", "u3"]
        loss = in_batch_negative_ce_loss(
            u, v, user_ids=user_ids, temperature=0.05
        )

        probs = torch.nn.functional.softmax(u @ v.t() / 0.05, dim=1)
        log_probs = torch.log(probs)
        per_sample = torch.stack([
            -(log_probs[0, 0] + log_probs[0, 1]),
            -(log_probs[1, 0] + log_probs[1, 1]),
            -log_probs[2, 2],
            -log_probs[3, 3],
        ])
        torch.testing.assert_close(loss, per_sample.mean())

    def test_duplicate_doc_across_users_is_positive(self):
        # different users clicked the same doc "a" (random batch, no dedup):
        # column 1 is doc "a" too, so it is a true positive for row 0
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        user_ids = ["u1", "u2", "u3", "u4"]
        video_ids = ["a", "a", "b", "c"]
        loss = in_batch_negative_ce_loss(
            u, v, user_ids=user_ids, video_ids=video_ids, temperature=0.05
        )

        probs = torch.nn.functional.softmax(u @ v.t() / 0.05, dim=1)
        log_probs = torch.log(probs)
        per_sample = torch.stack([
            -(log_probs[0, 0] + log_probs[0, 1]),
            -(log_probs[1, 0] + log_probs[1, 1]),
            -log_probs[2, 2],
            -log_probs[3, 3],
        ])
        torch.testing.assert_close(loss, per_sample.mean())

    def test_mask_is_union_of_user_and_video_rules(self):
        mask = build_positive_mask(
            user_ids=["u1", "u1", "u2", "u3"],
            video_ids=["a", "b", "a", "c"],
        )
        expected = torch.tensor([
            # row0: cols 0,1 same user u1; col 2 same doc "a"
            [True, True, True, False],
            [True, True, False, False],
            [True, False, True, False],
            [False, False, False, True],
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_positive_mask_overrides_ids(self):
        torch.manual_seed(0)
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        mask = torch.eye(BATCH, dtype=torch.bool)
        mask[0, 3] = True  # extra true positive from real click logs
        masked = in_batch_negative_ce_loss(u, v, positive_mask=mask)
        diagonal = in_batch_negative_ce_loss(u, v)
        # the extra positive adds -log p(u0, d3) > 0 to row 0's CE
        self.assertGreater(masked.item(), diagonal.item())

    def test_row_without_positive_rejected(self):
        u = torch.randn(BATCH, 8)
        bad = torch.zeros(BATCH, BATCH, dtype=torch.bool)
        bad[0, 0] = True  # rows 1..3 have no positive
        with self.assertRaises(ValueError):
            in_batch_negative_ce_loss(u, u, positive_mask=bad)


class TestNumericalStability(unittest.TestCase):
    def test_huge_logits_stay_finite(self):
        # unnormalized embeddings + tiny temperature -> raw logits ~1e7;
        # naive exp() would overflow to inf/nan without the max shift
        u = torch.randn(BATCH, 8) * 1000
        v = torch.randn(BATCH, 8) * 1000
        u.requires_grad_(True)
        loss = in_batch_negative_ce_loss(u, v, temperature=1e-4)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(u.grad).all())

    def test_shift_invariance(self):
        # the max-subtraction must not change the value vs normal inputs
        u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        logits = u @ v.t() / 0.05
        expected = torch.nn.functional.cross_entropy(logits, torch.arange(BATCH))
        actual = in_batch_negative_ce_loss(u, v)
        torch.testing.assert_close(actual, expected)

    def test_gradient_scales_inversely_with_temperature(self):
        """Not a numerical bug: dL/d(cosine) = (p - y)/temperature, so a
        sharper temperature mathematically amplifies gradients even though
        the softmax forward pass (max-shifted) is perfectly stable."""
        torch.manual_seed(0)
        base_u = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)
        base_v = torch.nn.functional.normalize(torch.randn(BATCH, 8), dim=-1)

        def grad_norm(temp):
            u = base_u.clone().requires_grad_(True)
            loss = in_batch_negative_ce_loss(u, base_v, temperature=temp)
            loss.backward()
            return u.grad.norm().item()

        sharp, soft = grad_norm(0.05), grad_norm(1.0)
        self.assertGreater(sharp, 5 * soft)  # ~1/temp amplification


if __name__ == "__main__":
    unittest.main()
