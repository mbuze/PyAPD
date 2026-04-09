"""Shared fixtures for PyAPD tests (CPU-only)."""

import math

import pytest
import torch

from PyAPD import apd_system

# ---------------------------------------------------------------------------
# Tiny APD systems
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_apd_2d():
    """N=5, D=2 APD on CPU with fixed seed."""
    s = apd_system(
        N=5, D=2, device="cpu", seed=42, error_tolerance=0.01, pixel_size_prefactor=2
    )
    s.assemble_pixels()
    return s


@pytest.fixture(scope="session")
def tiny_apd_3d():
    """N=5, D=3 APD on CPU with fixed seed."""
    return apd_system(
        N=5, D=3, device="cpu", seed=7, error_tolerance=0.05, pixel_size_prefactor=1
    )


@pytest.fixture(scope="session")
def tiny_masked_apd_2d():
    """5-grain 2D APD with top-half of pixels masked out."""
    s = apd_system(
        N=5, D=2, device="cpu", seed=42, error_tolerance=0.05, pixel_size_prefactor=2
    )
    s.assemble_pixels()
    mask = torch.zeros(s.pixel_params, dtype=torch.bool)
    mask[:, : s.pixel_params[1] // 2] = True  # keep lower half
    s.mask_pixels(mask)
    return s


@pytest.fixture(scope="session")
def tiny_arbitrary_apd_2d():
    """5-grain 2D APD with pixels sampled on a unit disk."""
    torch.manual_seed(99)
    M = 500
    # rejection-sample inside unit disk
    pts = []
    while len(pts) < M:
        p = torch.rand(2) * 2 - 1
        if p.norm() < 1:
            pts.append(p)
    Y = torch.stack(pts[:M])
    PS = torch.full((M,), math.pi / M)  # approx area of disk / M
    s = apd_system(N=5, D=2, device="cpu", seed=42, error_tolerance=0.05)
    s.set_pixels(Y, PS)
    return s


# ---------------------------------------------------------------------------
# Tiny grain map (used by log_res tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_grain_map(tiny_apd_2d):
    """Returns (Y, grain_map) tensors from the tiny 2D APD."""
    s = tiny_apd_2d
    grain_map = s.assemble_apd()
    return s.Y, grain_map
