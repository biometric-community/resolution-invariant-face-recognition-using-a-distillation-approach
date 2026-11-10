"""Teacher-student distillation across resolutions, on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rifr.data.celeba import downsample_to_scale
from rifr.eval import best_acc_eer
from rifr.models.student_teacher import FaceNet
from rifr.train import kl_div_logits


def test_student_shares_the_teachers_softmax_weights():
    """README: teacher and student share the Softmax classifier."""
    teacher, student = FaceNet(5, pretrained=False), FaceNet(5, pretrained=False)
    student.share_classifier_from(teacher)
    assert student.classifier.weight is teacher.classifier.weight


def test_distillation_loss_is_the_temperature_scaled_kl_divergence():
    torch.manual_seed(0)
    s, t, temp = torch.randn(4, 6), torch.randn(4, 6), 4.0
    p, q = F.softmax(s / temp, dim=1), F.softmax(t / temp, dim=1)
    expected = (q * (q.log() - p.log())).sum(dim=1).mean() * temp**2
    assert kl_div_logits(s, t, temp).item() == pytest.approx(expected.item(), rel=1e-5)
    assert kl_div_logits(t, t, temp).item() == pytest.approx(0.0, abs=1e-6)


def test_low_resolution_probes_keep_their_size_and_lose_detail():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 112, 112)
    low = downsample_to_scale(x, 0.25)
    assert low.shape == x.shape
    assert torch.equal(downsample_to_scale(x, 1.0), x)
    assert (low - x).abs().mean() > (downsample_to_scale(x, 0.5) - x).abs().mean() > 0


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)
