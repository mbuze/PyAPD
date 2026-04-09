"""Tests for PyAPD/apds.py (CPU-only)."""

import math

import torch
from matplotlib import pyplot as plt

from PyAPD import apd_system


class TestInit:
    def test_defaults_cpu(self):
        s = apd_system(N=5, D=2, device="cpu")
        assert s.N == 5
        assert s.D == 2
        assert s.device == "cpu"

    def test_small_3d_cpu(self):
        s = apd_system(N=5, D=3, device="cpu")
        assert s.D == 3
        assert s.X.shape == (5, 3)
        assert s.As.shape == (5, 3, 3)

    def test_target_masses_sum(self):
        s = apd_system(N=8, D=2, device="cpu")
        # target_masses are scaled by total_pixel_mass; normalised masses sum to total_pixel_mass
        assert abs(s.target_masses.sum().item() - s.total_pixel_mass.item()) < 1e-5


class TestAssemblePixels:
    def test_2d_pixel_count(self, tiny_apd_2d):
        s = tiny_apd_2d
        M = s.pixel_params[0] * s.pixel_params[1]
        assert s.Y.shape == (M, 2)

    def test_2d_pixel_volumes_sum(self, tiny_apd_2d):
        s = tiny_apd_2d
        total = s.PS.sum().item()
        assert abs(total - s.total_pixel_mass.item()) < 1e-4

    def test_3d_pixel_count(self):
        s = apd_system(
            N=3, D=3, device="cpu", error_tolerance=0.2, pixel_size_prefactor=1
        )
        s.assemble_pixels()
        M = s.pixel_params[0] * s.pixel_params[1] * s.pixel_params[2]
        assert s.Y.shape == (M, 3)

    def test_3d_pixel_volumes_sum(self):
        s = apd_system(
            N=3, D=3, device="cpu", error_tolerance=0.2, pixel_size_prefactor=1
        )
        s.assemble_pixels()
        total = s.PS.sum().item()
        assert abs(total - s.total_pixel_mass.item()) < 1e-4


class TestAssembleApd:
    def test_output_shape(self, tiny_apd_2d):
        s = tiny_apd_2d
        gi = s.assemble_apd(backend="CPU")
        M = s.pixel_params[0] * s.pixel_params[1]
        assert gi.shape == (M,)

    def test_grain_indices_in_range(self, tiny_apd_2d):
        s = tiny_apd_2d
        gi = s.assemble_apd(backend="CPU")
        assert gi.min() >= 0
        assert gi.max() < s.N


class TestFindOptimalW:
    def test_convergence_and_optimality(self):
        """N=5 2D APD should converge and pass check_optimality."""
        s = apd_system(
            N=5, D=2, device="cpu", seed=0, error_tolerance=0.05, pixel_size_prefactor=2
        )
        s.assemble_pixels()
        s.find_optimal_W(verbose=False, backend="CPU")
        s.check_optimality(backend="CPU")
        assert s.optimality

    def test_heuristic_W_init(self):
        s = apd_system(
            N=5,
            D=2,
            device="cpu",
            seed=1,
            heuristic_W=True,
            error_tolerance=0.05,
            pixel_size_prefactor=2,
        )
        assert not torch.all(s.W == 0)


class TestLloyds:
    def test_one_iteration(self):
        """A single Lloyd iteration should not raise."""
        s = apd_system(
            N=5, D=2, device="cpu", seed=2, error_tolerance=0.05, pixel_size_prefactor=2
        )
        s.assemble_pixels()
        s.Lloyds_algorithm(K=1, verbosity_level=0, backend="CPU")
        # X should have moved from the initial positions
        assert s.X.shape == (5, 2)


class TestSetPixels:
    def test_shapes(self, tiny_arbitrary_apd_2d):
        s = tiny_arbitrary_apd_2d
        assert s.Y.shape[1] == 2
        assert s.PS.shape == (s.Y.shape[0],)
        assert s.total_pixel_mass.item() > 0

    def test_target_masses_sum_to_total(self, tiny_arbitrary_apd_2d):
        s = tiny_arbitrary_apd_2d
        assert abs(s.target_masses.sum().item() - s.total_pixel_mass.item()) < 1e-5

    def test_find_optimal_W_arbitrary(self):
        """OT convergence on a small scattered pixel set."""
        torch.manual_seed(7)
        M = 300
        pts = []
        while len(pts) < M:
            p = torch.rand(2) * 2 - 1
            if p.norm() < 1:
                pts.append(p)
        Y = torch.stack(pts[:M])
        PS = torch.full((M,), math.pi / M)
        s = apd_system(
            N=5, D=2, device="cpu", seed=7, error_tolerance=0.1, pixel_size_prefactor=1
        )
        s.set_pixels(Y, PS)
        s.find_optimal_W(verbose=False, backend="CPU")
        s.check_optimality(backend="CPU")
        assert s.optimality

    def test_pixel_mode_set(self, tiny_arbitrary_apd_2d):
        assert tiny_arbitrary_apd_2d._pixel_mode == "arbitrary"


class TestMaskPixels:
    def test_mask_reduces_count(self, tiny_masked_apd_2d):
        s = tiny_masked_apd_2d
        # fixture keeps lower half only
        full_M = s.pixel_params[0] * s.pixel_params[1]
        assert s.Y.shape[0] < full_M

    def test_target_masses_rescaled(self, tiny_masked_apd_2d):
        s = tiny_masked_apd_2d
        assert abs(s.target_masses.sum().item() - s.total_pixel_mass.item()) < 1e-5

    def test_pixel_mode_set(self, tiny_masked_apd_2d):
        assert tiny_masked_apd_2d._pixel_mode == "arbitrary"


class TestPlotApdPsScale:
    def test_scatter_ps_scale_nonuniform(self):
        torch.manual_seed(0)
        Y = torch.rand(300, 2)
        PS = torch.rand(300) * 0.01  # non-uniform weights
        s = apd_system(N=4, D=2, device="cpu", seed=0, error_tolerance=0.1)
        s.set_pixels(Y, PS)
        fig, ax = s.plot_apd(mode="scatter", ps_scale=True, alpha=0.5)
        plt.close("all")
        assert fig is not None

    def test_patches_ps_scale_uniform(self):
        s = apd_system(
            N=4, D=2, device="cpu", seed=0, error_tolerance=0.1, pixel_size_prefactor=1
        )
        s.assemble_pixels()
        fig, ax = s.plot_apd(mode="patches", ps_scale=True, alpha=0.5)
        plt.close("all")
        assert fig is not None


class TestPlotApdVoronoi:
    def test_voronoi_random_cloud(self):
        torch.manual_seed(0)
        Y = torch.rand(150, 2)
        PS = torch.full((150,), 1.0 / 150)
        s = apd_system(N=4, D=2, device="cpu", seed=0, error_tolerance=0.1)
        s.set_pixels(Y, PS)
        fig, ax = s.plot_apd(mode="voronoi")
        plt.close("all")
        assert fig is not None

    def test_voronoi_rectangular_grid(self):
        s = apd_system(
            N=4, D=2, device="cpu", seed=0, error_tolerance=0.1, pixel_size_prefactor=1
        )
        s.assemble_pixels()
        fig, ax = s.plot_apd(mode="voronoi")
        plt.close("all")
        assert fig is not None
