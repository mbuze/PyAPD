"""Tests for PyAPD/log_res_utils.py (CPU-only)."""

import math

import pytest
import torch
from numpy.polynomial import Legendre, Polynomial

from PyAPD.log_res_utils import (
    assemble_design_matrix,
    calculate_moments_from_data,
    convert_from_lr_to_phys,
    convert_from_phys_to_lr,
    physical_heuristic_guess,
    reorder_variables,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grain_map(N=4, M=200, D=2, seed=0):
    torch.manual_seed(seed)
    Y = torch.rand(M, D)
    grain_map = torch.randint(0, N, (M,))
    return Y, grain_map


# ---------------------------------------------------------------------------
# assemble_design_matrix
# ---------------------------------------------------------------------------


class TestAssembleDesignMatrix:
    @pytest.mark.parametrize("D,ho", [(2, 1), (2, 2), (3, 1), (3, 2)])
    def test_shape(self, D, ho):
        M = 50
        Y = torch.rand(M, D)
        dm = assemble_design_matrix(Y, ho=ho, basis=Polynomial)
        K = math.comb(D + ho, ho)
        assert dm.shape == (M, K)

    def test_legendre_basis_shape(self):
        M, D, ho = 40, 2, 2
        Y = torch.rand(M, D)
        dm = assemble_design_matrix(Y, ho=ho, basis=Legendre)
        K = math.comb(D + ho, ho)
        assert dm.shape == (M, K)

    def test_constant_column(self):
        """First column should be constant (degree-0 basis element = 1)."""
        M, D, ho = 30, 2, 2
        Y = torch.rand(M, D)
        dm = assemble_design_matrix(Y, ho=ho, basis=Polynomial)
        # With Polynomial basis, degree-0 = 1 everywhere → first col is 1/eps
        eps = 1.0
        assert torch.allclose(dm[:, 0], torch.ones(M) / eps, atol=1e-5)


# ---------------------------------------------------------------------------
# reorder_variables
# ---------------------------------------------------------------------------


class TestReorderVariables:
    def test_output_shape(self):
        N, D, ho_start, ho_end = 5, 2, 1, 2
        K_start = math.comb(D + ho_start, ho_start)
        K_end = math.comb(D + ho_end, ho_end)
        theta = torch.randn(N, K_start)
        theta_new = reorder_variables(theta, D, ho_start, ho_end)
        assert theta_new.shape == (N, K_end)

    def test_existing_values_preserved(self):
        """Values from ho_start columns must appear unchanged in ho_end result."""
        N, D, ho_start, ho_end = 3, 2, 1, 3
        K_start = math.comb(D + ho_start, ho_start)
        theta = torch.randn(N, K_start)
        theta_new = reorder_variables(theta, D, ho_start, ho_end)
        # The K_start entries that were non-zero should still match theta
        import itertools

        I_start = [
            idx
            for idx in itertools.product(range(ho_start + 1), repeat=D)
            if sum(idx) < ho_start + 1
        ]
        I_end = [
            idx
            for idx in itertools.product(range(ho_end + 1), repeat=D)
            if sum(idx) < ho_end + 1
        ]
        positions = [I_end.index(i) for i in I_start]
        assert torch.allclose(theta_new[:, positions], theta)

    def test_zero_padding(self):
        """New columns (beyond K_start) must be zero-initialised."""
        N, D, ho_start, ho_end = 4, 2, 1, 2
        K_start = math.comb(D + ho_start, ho_start)
        theta = torch.randn(N, K_start)
        theta_new = reorder_variables(theta, D, ho_start, ho_end)
        import itertools

        I_start = [
            idx
            for idx in itertools.product(range(ho_start + 1), repeat=D)
            if sum(idx) < ho_start + 1
        ]
        I_end = [
            idx
            for idx in itertools.product(range(ho_end + 1), repeat=D)
            if sum(idx) < ho_end + 1
        ]
        new_positions = [i for i, idx in enumerate(I_end) if idx not in I_start]
        assert torch.all(theta_new[:, new_positions] == 0)


# ---------------------------------------------------------------------------
# convert_from_phys_to_lr / convert_from_lr_to_phys
# ---------------------------------------------------------------------------


class TestConvertPhysLr:
    def test_roundtrip_ho1(self):
        """lr → phys → lr should be identity (ho=1)."""
        N = 6
        theta = torch.randn(N, math.comb(2 + 1, 1))
        phys = convert_from_lr_to_phys(theta, ho=1)
        theta_back = convert_from_phys_to_lr(phys, ho=1)
        assert torch.allclose(theta, theta_back, atol=1e-5)

    def test_roundtrip_ho2(self):
        """phys → lr → phys should be identity (ho=2, D=2)."""
        N = 4
        # Build a valid physical guess from a small grain map
        Y, grain_map = _make_grain_map(N=N, M=400, D=2)
        phys = physical_heuristic_guess(grain_map, Y, ho=2)
        lr = convert_from_phys_to_lr(phys, ho=2)
        phys_back = convert_from_lr_to_phys(lr, ho=2)
        assert torch.allclose(phys, phys_back, atol=1e-4)


# ---------------------------------------------------------------------------
# calculate_moments_from_data
# ---------------------------------------------------------------------------


class TestCalculateMoments:
    @pytest.mark.parametrize("D,ho", [(2, 1), (2, 2)])
    def test_shape(self, D, ho):
        N = 4
        Y, grain_map = _make_grain_map(N=N, M=300, D=D)
        moments = calculate_moments_from_data(grain_map, Y, ho=ho, basis=Polynomial)
        K = math.comb(D + ho, ho)
        assert moments.shape == (N, K)

    def test_zeroth_moment_is_count(self):
        """First column of moments (degree-0) is pixel count per grain."""
        N, M, D = 3, 600, 2
        Y, grain_map = _make_grain_map(N=N, M=M, D=D)
        moments = calculate_moments_from_data(grain_map, Y, ho=1, basis=Polynomial)
        counts = torch.bincount(grain_map, minlength=N).float()
        assert torch.allclose(moments[:, 0], counts, atol=1e-4)
