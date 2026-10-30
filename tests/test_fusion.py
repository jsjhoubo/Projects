"""Tests for ttsn/fusion.py — big layer 2 (fusion + BasicTTSN assembly).

What is covered:
- ConcatMLPFusion: fuse(a, b) -> [B, out_dim]; building-block stacking
  (output of one block feeds the next)
- BasicTTSN forward: (user_info, video_info, interactive_info) ->
  two [B, output_dim] embeddings
- BasicTTSN training_step: autograd on, loss backprops into all three
  encoders and both fusion MLPs; per-sample weights accepted
- BasicTTSN infer/score: inference_mode (no grad), L2-normalized outputs,
  score is the pairwise cosine [B]
"""

import unittest

import torch

from ttsn.fusion import BasicTTSN, ConcatMLPFusion
from tests.test_encoding import make_interaction, make_user, make_video

B = 4


class TestConcatMLPFusion(unittest.TestCase):
    def test_fuse_shape(self):
        """fuse(a[B,64], b[B,112]) -> [B, 128]."""
        block = ConcatMLPFusion(64, 112, 128)
        out = block(torch.randn(B, 64), torch.randn(B, 112))
        self.assertEqual(out.shape, (B, 128))

    def test_blocks_stack(self):
        """Blocks share the fuse(a, b) interface, so they chain:
        fuse(fuse(a, b), c) -> one final embedding."""
        f1 = ConcatMLPFusion(32, 16, 64)
        f2 = ConcatMLPFusion(64, 8, 128)
        out = f2(f1(torch.randn(B, 32), torch.randn(B, 16)), torch.randn(B, 8))
        self.assertEqual(out.shape, (B, 128))


class TestBasicTTSNForward(unittest.TestCase):
    def test_forward_returns_two_embeddings(self):
        """forward consumes the three info objects and returns
        (user_emb, video_emb), both [B, output_dim]."""
        model = BasicTTSN()
        u, v = model(make_user(), make_video(), make_interaction())
        self.assertEqual(u.shape, (B, 128))
        self.assertEqual(v.shape, (B, 128))


class TestBasicTTSNTraining(unittest.TestCase):
    def test_training_step_backprops_everywhere(self):
        """training_step builds a graph (even under no_grad) and the loss
        gradient reaches user/video/interaction encoders and both fusions."""
        model = BasicTTSN()
        with torch.no_grad():  # training_step must re-enable autograd
            loss = model.training_step(
                make_user(), make_video(), make_interaction(),
                sample_weights=torch.tensor([1.0, 2.0, 1.0, 0.5]),
            )
        self.assertTrue(loss.requires_grad)
        loss.backward()
        for module in (model.user_encoder, model.video_encoder,
                       model.interaction_encoder, model.user_fusion,
                       model.video_fusion):
            self.assertTrue(
                any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in module.parameters()),
                f"no gradient reached {type(module).__name__}",
            )

    def test_one_sgd_step_reduces_loss(self):
        """A single optimizer step on one batch lowers the training loss."""
        torch.manual_seed(0)
        user, video, inter = make_user(), make_video(), make_interaction()
        model = BasicTTSN()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        before = model.training_step(user, video, inter)
        before.backward()
        opt.step()
        after = model.training_step(user, video, inter)
        self.assertLess(after.item(), before.item())


class TestBasicTTSNInference(unittest.TestCase):
    def test_infer_no_grad_and_normalized(self):
        """infer runs without autograd and returns L2-normalized embeddings."""
        model = BasicTTSN()
        u, v = model.infer(make_user(), make_video(), make_interaction())
        self.assertFalse(u.requires_grad)
        self.assertFalse(v.requires_grad)
        torch.testing.assert_close(u.norm(dim=-1), torch.ones(B))
        torch.testing.assert_close(v.norm(dim=-1), torch.ones(B))
        self.assertFalse(model.training)  # eval mode after infer

    def test_score_is_pairwise_cosine(self):
        """score returns [B] = cosine(user_i, video_i), within [-1, 1]."""
        model = BasicTTSN()
        s = model.score(make_user(), make_video(), make_interaction())
        self.assertEqual(s.shape, (B,))
        self.assertTrue((s.abs() <= 1.0 + 1e-5).all())


if __name__ == "__main__":
    unittest.main()
