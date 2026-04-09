"""Tests for PyAPD/geometry_utils.py (CPU-only)."""

import torch

from PyAPD.geometry_utils import (
    initial_guess_heuristic,
    sample_normalised_spd_matrices,
    sample_seeds_with_exclusion,
    specify_volumes,
)


class TestSampleSeedsWithExclusion:
    def test_shape_2d(self):
        X = sample_seeds_with_exclusion(8, dim=2)
        assert X.shape == (8, 2)

    def test_shape_3d(self):
        X = sample_seeds_with_exclusion(6, dim=3)
        assert X.shape == (6, 3)

    def test_exclusion_radius(self):
        n, dim = 10, 2
        radius_prefactor = 0.1
        radius = radius_prefactor * n ** (-1 / dim)
        X = sample_seeds_with_exclusion(n, dim=dim, radius_prefactor=radius_prefactor)
        # All pairwise distances must exceed the exclusion radius
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                assert torch.norm(X[i] - X[j]) > radius

    def test_reproducibility(self):
        torch.manual_seed(0)
        X1 = sample_seeds_with_exclusion(5, dim=2)
        torch.manual_seed(0)
        X2 = sample_seeds_with_exclusion(5, dim=2)
        assert torch.allclose(X1, X2)


class TestSampleNormalisedSpdMatrices:
    def test_det_2d(self):
        As = sample_normalised_spd_matrices(20, dim=2, ani_thres=0.3)
        dets = torch.linalg.det(As)
        assert torch.allclose(dets, torch.ones(20), atol=1e-5)

    def test_det_3d(self):
        As = sample_normalised_spd_matrices(10, dim=3, ani_thres=0.3)
        dets = torch.linalg.det(As)
        assert torch.allclose(dets, torch.ones(10), atol=1e-5)

    def test_symmetry_2d(self):
        As = sample_normalised_spd_matrices(8, dim=2)
        assert torch.allclose(As, As.mT, atol=1e-6)

    def test_positive_definite_2d(self):
        As = sample_normalised_spd_matrices(8, dim=2)
        eigvals = torch.linalg.eigvalsh(As)
        assert (eigvals > 0).all()


class TestSpecifyVolumes:
    def test_sums_to_one(self):
        v = specify_volumes(20)
        assert abs(v.sum().item() - 1.0) < 1e-6

    def test_nonnegative(self):
        v = specify_volumes(15)
        assert (v >= 0).all()

    def test_shape(self):
        v = specify_volumes(7)
        assert v.shape == (7,)


class TestInitialGuessHeuristic:
    def test_shape_2d(self):
        As = sample_normalised_spd_matrices(5, dim=2)
        TVs = specify_volumes(5)
        W = initial_guess_heuristic(As, TVs, D=2)
        assert W.shape == (5,)

    def test_positive(self):
        As = sample_normalised_spd_matrices(5, dim=2)
        TVs = specify_volumes(5)
        W = initial_guess_heuristic(As, TVs, D=2)
        assert (W > 0).all()
