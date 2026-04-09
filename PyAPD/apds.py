import time

import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
from torchmin import minimize as minimize_torch

from . import (
    override_optimality_condition,  # noqa: F401 -- side-effect: patches torchmin
)
from .geometry_utils import (
    initial_guess_heuristic,
    sample_normalised_spd_matrices,
    sample_seeds_with_exclusion,
    sample_spd_matrices_perturbed_from_identity,
    specify_volumes,
)


class apd_system:
    """
    An anisotropic power diagram system.
    """

    def __init__(
        self,
        domain=None,
        X=None,
        As=None,
        W=None,
        target_masses=None,
        pixel_params=None,
        dt=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # convenience constructor options:
        N=10,
        D=2,
        ani_thres=0.25,
        heuristic_W=False,
        radius_of_exclusion=0.01,
        det_constraint=True,
        error_tolerance=0.01,
        pixel_size_prefactor=2,
        seed=-1,
    ):
        """
        Construct an anisotropic power diagram system.

        Parameters
        ----------
        domain : Tensor, shape (D, 2), optional
            Row i gives [lower, upper] bound for dimension i.
            Defaults to the unit hypercube [0, 1]^D.
        X : Tensor, shape (N, D), optional
            Seed point positions. Sampled with exclusion if not provided.
        As : Tensor, shape (N, D, D), optional
            Per-grain SPD anisotropy matrices. Sampled if not provided.
        W : Tensor, shape (N,), optional
            Per-grain weights. Zero-initialised if not provided.
        target_masses : Tensor, shape (N,), optional
            Target grain volumes summing to 1. Drawn randomly if not provided.
        pixel_params : tuple of int, optional
            Number of pixels along each spatial dimension. Computed from
            `error_tolerance` and `pixel_size_prefactor` if not provided.
        dt : torch.dtype
            Floating-point dtype. Default: ``torch.float32``.
        device : str
            Compute device. Default: CUDA if available, else CPU.
        N : int
            Number of grains (used only when X is not provided). Default: 10.
        D : int
            Spatial dimension (used only when X / domain not provided). Default: 2.
        ani_thres : float
            Anisotropy threshold in [0, 1] passed to the matrix sampler.
            Default: 0.25.
        heuristic_W : bool
            Initialise W with the Teferra-Rowenhorst heuristic. Default: False.
        radius_of_exclusion : float
            Exclusion-radius prefactor for seed sampling. Default: 0.01.
        det_constraint : bool
            Enforce det(A) = 1 when sampling anisotropy matrices. Default: True.
        error_tolerance : float
            Relative volume-error tolerance used in optimality checks and to set
            the default pixel resolution. Default: 0.01.
        pixel_size_prefactor : int
            Multiplier applied to the computed pixel count. Default: 2.
        seed : int
            Manual random seed (< 0 means no seed is set). Default: -1.
        """

        self.N = N
        self.D = D

        self.dt = dt
        self.device = device
        self.error_tolerance = error_tolerance
        self.radius_of_exclusion = radius_of_exclusion
        self.ani_thres = ani_thres
        self.seed = seed
        self.det_constraint = det_constraint
        self.heuristic_W = heuristic_W
        self.pixel_size_prefactor = pixel_size_prefactor

        self.set_domain(domain)

        self.set_X(X)

        self.set_As(As)
        self.set_target_masses(target_masses)

        self.set_W(W)

        self.set_pixel_params(pixel_params)

        self._pixel_mode = None  # 'rectangular' | 'arbitrary'
        self.Y = None
        self.y = None
        self.PS = None
        self.optimality = False
        self.data = {}

    def set_domain(self, domain=None):
        """
        Set the spatial domain.

        Parameters
        ----------
        domain : Tensor, shape (D, 2), optional
            Row i gives [lower, upper] bound for dimension i.
            Defaults to the unit hypercube [0, 1]^D.
        """
        if domain is None:
            domain = torch.tensor([[0, 1]] * self.D)
        else:
            self.D = domain.shape[0]

        self.domain = domain.to(device=self.device, dtype=self.dt)
        self.total_pixel_mass = torch.prod(self.domain[:, 1] - self.domain[:, 0]).to(
            device=self.device, dtype=self.dt
        )

    def set_X(self, X=None, verbose=False):
        """
        Set the grain seed positions.

        Parameters
        ----------
        X : Tensor, shape (N, D), optional
            Seed positions. If not provided, N points are sampled uniformly
            with mutual exclusion inside [0, 1]^D and then mapped to `domain`.
        verbose : bool
            Print rejection statistics during seed sampling. Default: False.
        """
        self._X_auto = X is None
        if X is None:
            if not (self.seed == -1):
                torch.manual_seed(self.seed)

            X = sample_seeds_with_exclusion(
                self.N,
                dim=self.D,
                radius_prefactor=self.radius_of_exclusion,
                verbose=verbose,
            ).to(device=self.device, dtype=self.dt)
            X = self.domain[:, 0] + (self.domain[:, 1] - self.domain[:, 0]) * X
        else:
            self.N = len(X)
            self.D = X.shape[-1]

        self.X = X.to(device=self.device, dtype=self.dt)
        self.x = LazyTensor(self.X.view(self.N, 1, self.D))  # (N, 1, D)

    def set_As(self, As=None):
        """
        Set the per-grain anisotropy matrices.

        Parameters
        ----------
        As : Tensor, shape (N, D, D), optional
            SPD anisotropy matrices. If not provided, N matrices are sampled:
            normalised (det = 1) when `det_constraint` is True, perturbed from
            identity otherwise.
        """
        if As is None:
            if not (self.seed == -1):
                torch.manual_seed(100 + self.seed)
            if self.det_constraint:
                As = sample_normalised_spd_matrices(
                    self.N, dim=self.D, ani_thres=self.ani_thres
                )
            else:
                As = sample_spd_matrices_perturbed_from_identity(
                    self.N, dim=self.D, amp=self.ani_thres
                )

        self.As = As.to(device=self.device, dtype=self.dt)
        self.a = LazyTensor(self.As.view(self.N, 1, self.D * self.D))

    def set_target_masses(self, target_masses=None):
        """
        Set the per-grain target masses (volumes).

        Parameters
        ----------
        target_masses : Tensor, shape (N,), optional
            Target volumes summing to 1. If not provided, volumes are drawn
            randomly via `specify_volumes` and scaled by `total_pixel_mass`.
        """
        if target_masses is None:
            target_masses = specify_volumes(self.N)

        fracs = target_masses.to(device=self.device, dtype=self.dt)
        self._target_fractions = fracs / fracs.sum()
        self.target_masses = self._target_fractions * self.total_pixel_mass

    def set_W(self, W=None):
        """
        Set the per-grain weights.

        Parameters
        ----------
        W : Tensor, shape (N,), optional
            Power-diagram weights. Defaults to zero (uniform Voronoi). When
            `heuristic_W` is True, the Teferra-Rowenhorst heuristic is used
            instead.
        """
        self._W_from_heuristic = W is None and self.heuristic_W
        if W is None:
            if self.heuristic_W:
                W = initial_guess_heuristic(self.As, self.target_masses, self.D)
            else:
                W = torch.zeros(self.N)

        self.W = W.to(device=self.device, dtype=self.dt)
        self.w = LazyTensor(self.W.view(self.N, 1, 1))

    def set_pixel_params(self, pixel_params=None, verbose=False):
        """
        Set the pixel grid resolution.

        Parameters
        ----------
        pixel_params : tuple of int, optional
            Number of pixels per spatial dimension, e.g. ``(M, M)`` for 2D.
            If not provided, computed from `error_tolerance` and
            `pixel_size_prefactor` so that the smallest grain is resolved.
        verbose : bool
            Print the computed pixel count. Default: False.
        """
        if pixel_params is None:
            M = (
                self.total_pixel_mass
                / (self.error_tolerance * torch.min(self.target_masses))
            ) ** (1 / self.D)
            M = int(self.pixel_size_prefactor * int(M.item()))
            pixel_params = (M,) * self.D
        self.pixel_params = pixel_params
        if verbose:
            print("M = ", pixel_params)

    def assemble_pixels(self):
        """
        Assemble the pixels/voxels on a uniform rectangular grid.
        """
        i = 0
        grid_x = torch.linspace(
            self.domain[i, 0]
            + 0.5 * (self.domain[i, 1] - self.domain[i, 0]) / self.pixel_params[i],
            self.domain[i, 1]
            - 0.5 * (self.domain[i, 1] - self.domain[i, 0]) / self.pixel_params[i],
            self.pixel_params[i],
        )
        i = 1
        grid_y = torch.linspace(
            self.domain[i, 0]
            + 0.5 * (self.domain[i, 1] - self.domain[i, 0]) / self.pixel_params[i],
            self.domain[i, 1]
            - 0.5 * (self.domain[i, 1] - self.domain[i, 0]) / self.pixel_params[i],
            self.pixel_params[i],
        )
        if self.D == 3:
            i = 2
            grid_z = torch.linspace(
                self.domain[i, 0]
                + 0.5 * (self.domain[i, 1] - self.domain[i, 0]) / self.pixel_params[i],
                self.domain[i, 1]
                - 0.5 * (self.domain[i, 1] - self.domain[i, 0]) / self.pixel_params[i],
                self.pixel_params[i],
            )

        mesh = (
            torch.meshgrid((grid_x, grid_y), indexing="ij")
            if self.D == 2
            else torch.meshgrid((grid_x, grid_y, grid_z), indexing="ij")
        )

        pixels = torch.stack(mesh, dim=-1).to(device=self.device, dtype=self.dt)
        pixels = pixels.reshape(-1, self.D)

        pixel_vols = (self.domain[:, 1] - self.domain[:, 0]) / torch.tensor(
            self.pixel_params
        ).to(device=self.device, dtype=self.dt)

        i = 0
        PS_x = pixel_vols[i] * torch.ones(self.pixel_params[i]).to(
            device=self.device, dtype=self.dt
        )
        i = 1
        PS_y = pixel_vols[i] * torch.ones(self.pixel_params[i]).to(
            device=self.device, dtype=self.dt
        )

        if self.D == 3:
            i = 2
            PS_z = pixel_vols[i] * torch.ones(self.pixel_params[i]).to(
                device=self.device, dtype=self.dt
            )

        PS = PS_x[:, None] @ PS_y[None, :]
        if self.D == 3:
            PS = PS[:, :, None] @ PS_z[None, :]

        PS = PS.reshape(-1, 1).flatten()
        self.Y = pixels
        self.PS = PS
        self.y = LazyTensor(
            self.Y.view(1, torch.prod(torch.tensor(self.pixel_params)).item(), self.D)
        )
        self.total_pixel_mass = self.PS.sum()
        self.target_masses = self._target_fractions * self.total_pixel_mass
        self._pixel_mode = "rectangular"
        if self._W_from_heuristic:
            self.W = initial_guess_heuristic(self.As, self.target_masses, self.D)
            self.w = LazyTensor(self.W.view(self.N, 1, 1))

    def set_pixels(self, Y, PS):
        """
        Set arbitrary pixel positions and masses.

        Parameters
        ----------
        Y  : Tensor, shape (M, D) — pixel/voxel positions.
        PS : Tensor, shape (M,)   — pixel measure (area in 2D, volume in 3D).
        """
        M = Y.shape[0]
        self.Y = Y.to(device=self.device, dtype=self.dt)
        self.PS = PS.to(device=self.device, dtype=self.dt)
        self.total_pixel_mass = self.PS.sum()
        self.target_masses = self._target_fractions * self.total_pixel_mass
        self.y = LazyTensor(self.Y.view(1, M, self.D))
        self._pixel_mode = "arbitrary"
        # Resample seeds within bounding box of Y if they were auto-generated,
        # since the default sampling maps to domain which may not overlap Y.
        if self._X_auto:
            lb = self.Y.min(dim=0).values
            ub = self.Y.max(dim=0).values
            if not (self.seed == -1):
                torch.manual_seed(self.seed)
            X_raw = sample_seeds_with_exclusion(
                self.N,
                dim=self.D,
                radius_prefactor=self.radius_of_exclusion,
            ).to(device=self.device, dtype=self.dt)
            self.X = lb + (ub - lb) * X_raw
            self.x = LazyTensor(self.X.view(self.N, 1, self.D))
        if self._W_from_heuristic:
            self.W = initial_guess_heuristic(self.As, self.target_masses, self.D)
            self.w = LazyTensor(self.W.view(self.N, 1, 1))

    def mask_pixels(self, mask):
        """
        Remove pixels using a boolean mask.  Call after assemble_pixels().

        Parameters
        ----------
        mask : BoolTensor, shape pixel_params or (M,) — True = keep.
        """
        mask_flat = mask.reshape(-1).to(device=self.device)
        self.Y = self.Y[mask_flat]
        self.PS = self.PS[mask_flat]
        self.total_pixel_mass = self.PS.sum()
        self.target_masses = self._target_fractions * self.total_pixel_mass
        M = self.Y.shape[0]
        self.y = LazyTensor(self.Y.view(1, M, self.D))
        self._pixel_mode = "arbitrary"
        if self._W_from_heuristic:
            self.W = initial_guess_heuristic(self.As, self.target_masses, self.D)
            self.w = LazyTensor(self.W.view(self.N, 1, 1))

    def assemble_apd(
        self, record_time=False, verbose=False, color_by=None, backend="auto"
    ):
        """
        Assemble the apd by finding which grain each pixel belongs to.
        """
        if self.Y is None:
            self.assemble_pixels()
        start = time.time()
        D_ij = ((self.y - self.x) | self.a.matvecmult(self.y - self.x)) - self.w
        # Find which grain each pixel belongs to
        grain_indices = D_ij.argmin(dim=0, backend=backend).ravel()
        time_taken = time.time() - start
        if record_time:
            self.apd_gen_time = time_taken

        if verbose:
            print("APD generated in:", time_taken, "seconds.")
        if color_by is not None:
            return color_by[grain_indices]
        else:
            return grain_indices

    def plot_apd(
        self, color_by=None, mode="auto", alpha=None, ps_scale=False, marker_scale=20.0
    ):
        """
        Plot the APD (2D only).

        Parameters
        ----------
        color_by : Tensor, shape (N,), optional
            Per-grain values used for colouring. Defaults to grain indices.
        mode : str
            Rendering mode: 'auto', 'grid', 'scatter', 'interpolated', 'patches',
            'voronoi'.
            'auto' selects 'grid' for rectangular pixels and 'interpolated' for
            arbitrary pixel clouds.
            'voronoi' renders each pixel as its Voronoi cell — geometrically correct
            (cell area = PS) when the pixel set comes from a tessellation, e.g. a
            regular hex grid or a uniform rectangular grid. For variable-spacing grids
            render manually using the known boundary arrays; for non-uniform PS clouds
            use 'scatter' + ps_scale.
        alpha : float or None, optional
            Transparency in [0, 1]. None = fully opaque. Default: None.
        ps_scale : bool, optional
            Scale marker/patch sizes proportionally to PS. Default: False.
        marker_scale : float, optional
            Reference marker size (matplotlib points²) for an average-PS point
            in scatter mode when ps_scale=True. Default: 20.0.
        """
        if self.D != 2:
            return self.assemble_apd(color_by=color_by)

        assignments = self.assemble_apd(color_by=color_by)

        if mode == "auto":
            mode = "grid" if self._pixel_mode == "rectangular" else "interpolated"

        if mode == "grid":
            img = assignments.reshape(self.pixel_params).transpose(0, 1).cpu()
            fig, ax = plt.subplots(1, 1, figsize=(10.5, 10.5))
            imshow_kw = dict(origin="lower", extent=torch.flatten(self.domain).tolist())
            if alpha is not None:
                imshow_kw["alpha"] = alpha
            ax.imshow(img, **imshow_kw)

        elif mode == "scatter":
            import numpy as np

            fig, ax = plt.subplots(1, 1, figsize=(10.5, 10.5))
            ps_np = self.PS.cpu().numpy()
            if ps_scale:
                s_vals = marker_scale * ps_np / ps_np.mean()
            else:
                s_vals = 1
            scatter_kw = dict(c=assignments.cpu(), s=s_vals, cmap="tab20")
            if alpha is not None:
                scatter_kw["alpha"] = alpha
            ax.scatter(self.Y[:, 0].cpu(), self.Y[:, 1].cpu(), **scatter_kw)

        elif mode == "interpolated":
            from scipy.interpolate import griddata

            Y_np = self.Y.cpu().numpy()
            vals = assignments.cpu().numpy()
            xi = torch.linspace(float(Y_np[:, 0].min()), float(Y_np[:, 0].max()), 512)
            yi = torch.linspace(float(Y_np[:, 1].min()), float(Y_np[:, 1].max()), 512)
            xi_g, yi_g = torch.meshgrid(xi, yi, indexing="ij")
            img = griddata(Y_np, vals, (xi_g.numpy(), yi_g.numpy()), method="nearest")
            fig, ax = plt.subplots(1, 1, figsize=(10.5, 10.5))
            imshow_kw = dict(
                origin="lower",
                extent=[float(xi[0]), float(xi[-1]), float(yi[0]), float(yi[-1])],
            )
            if alpha is not None:
                imshow_kw["alpha"] = alpha
            ax.imshow(img.T, **imshow_kw)

        elif mode == "patches":
            import numpy as np
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Rectangle

            Y_np = self.Y.cpu().numpy()
            vals = assignments.cpu().numpy()
            # infer pixel half-size from nearest-neighbour distances (approximate)
            hw = float(np.median(np.diff(np.unique(Y_np[:, 0])))) / 2
            if ps_scale:
                ps_np = self.PS.cpu().numpy()
                hw_vals = hw * np.sqrt(ps_np / ps_np.mean())
            else:
                hw_vals = np.full(len(Y_np), hw)
            patches = [
                Rectangle(
                    (y[0] - hw_vals[i], y[1] - hw_vals[i]),
                    2 * hw_vals[i],
                    2 * hw_vals[i],
                )
                for i, y in enumerate(Y_np)
            ]
            pc = PatchCollection(patches, array=vals, cmap="tab20")
            if alpha is not None:
                pc.set_alpha(alpha)
            fig, ax = plt.subplots(1, 1, figsize=(10.5, 10.5))
            ax.add_collection(pc)
            ax.autoscale()

        elif mode == "voronoi":
            import numpy as np
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Polygon
            from scipy.spatial import Voronoi

            Y_np = self.Y.cpu().numpy()
            vals = assignments.cpu().numpy().astype(float)

            # Mirror-point trick: reflect Y across each side of an enlarged bounding box
            # so that every original pixel has a finite (bounded) Voronoi region.
            x_min, y_min = Y_np.min(axis=0)
            x_max, y_max = Y_np.max(axis=0)
            m = 0.05 * max(x_max - x_min, y_max - y_min)
            bx0, bx1 = x_min - m, x_max + m
            by0, by1 = y_min - m, y_max + m
            mirror = np.vstack(
                [
                    Y_np,
                    np.c_[2 * bx0 - Y_np[:, 0], Y_np[:, 1]],  # left mirror
                    np.c_[2 * bx1 - Y_np[:, 0], Y_np[:, 1]],  # right mirror
                    np.c_[Y_np[:, 0], 2 * by0 - Y_np[:, 1]],  # bottom mirror
                    np.c_[Y_np[:, 0], 2 * by1 - Y_np[:, 1]],  # top mirror
                ]
            )
            vor = Voronoi(mirror)

            polygons, poly_vals = [], []
            for i in range(len(Y_np)):
                region = vor.regions[vor.point_region[i]]
                if -1 in region or len(region) == 0:
                    continue  # skip degenerate cells (should not occur with mirror trick)
                polygons.append(Polygon(vor.vertices[region]))
                poly_vals.append(vals[i])

            pc = PatchCollection(polygons, array=np.asarray(poly_vals), cmap="tab20")
            if alpha is not None:
                pc.set_alpha(alpha)
            fig, ax = plt.subplots(1, 1, figsize=(10.5, 10.5))
            ax.add_collection(pc)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        return fig, ax

    def plot_ellipses(self):
        if self.D == 2:
            decomp = torch.linalg.eigh(self.As)
            AB = decomp.eigenvalues ** (-0.5)
            scaling = (self.target_masses / torch.pi) ** (1 / 2)
            AB[:, 0] = AB[:, 0] * scaling
            AB[:, 1] = AB[:, 1] * scaling
            Rots = decomp.eigenvectors
            t = torch.linspace(0, 2 * torch.pi, 80).to(
                device=self.device, dtype=self.dt
            )
            Ell1 = AB[:, 0, None] @ torch.cos(t)[None, :]
            Ell2 = AB[:, 1, None] @ torch.sin(t)[None, :]

            Ell = torch.stack([Ell1, Ell2])

            Ell = (Ell.transpose(0, 2)).transpose(0, 1)
            Ell_rot = Ell @ Rots
            Ell_rot_shifted = [Ell_rot[i] + self.X[i] for i in range(len(Ell_rot))]
            fig, (ax1) = plt.subplots(1, 1)
            fig.set_size_inches(10.5, 10.5, forward=True)
            for i in range(0, len(Ell_rot_shifted)):
                ax1.scatter(self.X.cpu()[i, 0], self.X.cpu()[i, 1], c="r", s=3)
                ax1.plot(
                    Ell_rot_shifted[i].cpu()[:, 0],
                    Ell_rot_shifted[i].cpu()[:, 1],
                    c="k",
                )
            if self.Y is not None:
                ax1.set_xlim(self.Y[:, 0].min().item(), self.Y[:, 0].max().item())
                ax1.set_ylim(self.Y[:, 1].min().item(), self.Y[:, 1].max().item())
            else:
                ax1.set_xlim(self.domain[0].cpu().numpy())
                ax1.set_ylim(self.domain[1].cpu().numpy())
            return fig, ax1

    def OT_dual_function(self, W, backend="auto"):
        """
        Helper function for assembling the OT dual function g(W).
        """
        self.W = W
        self.w = LazyTensor(self.W.view(self.N, 1, 1))

        D_ij = ((self.y - self.x) | self.a.matvecmult(self.y - self.x)) - self.w
        idx = D_ij.argmin(dim=0, backend=backend).view(-1)

        ind_select = torch.index_select(self.X, 0, idx) - self.Y
        MV = torch.einsum("bij,bj->bi", torch.index_select(self.As, 0, idx), ind_select)
        sD_ij = torch.einsum("bj,bj->b", MV, ind_select) - torch.index_select(
            self.W, 0, idx
        )
        g = -torch.dot(self.target_masses, self.W) - torch.dot(sD_ij, self.PS)
        return g

    def check_optimality(
        self, error_wrt_each_grain=True, return_gradient_and_error=False, backend="auto"
    ):
        """
        Check whether the APD generated by (X,As,W) is optimal with respect to target masses.
        """
        if self.Y is None:
            self.assemble_pixels()

        D_ij = ((self.y - self.x) | self.a.matvecmult(self.y - self.x)) - self.w

        grain_indices = D_ij.argmin(dim=0, backend=backend).ravel()
        volumes = torch.bincount(grain_indices, self.PS, minlength=self.N)
        Dg = volumes - self.target_masses
        if error_wrt_each_grain:
            error = torch.max(torch.abs(Dg) / self.target_masses)
        else:
            error = torch.max(torch.abs(Dg) / torch.min(self.target_masses))

        if error < self.error_tolerance + 1e-5:
            print("The APD is optimal!")
            print("Percentage error = ", (100 * error).item())
            self.optimality = True
        else:
            print("Precision loss detected!")
            print("Percentage error = ", (100 * error).item())
        if return_gradient_and_error:
            return Dg, error

    def find_optimal_W(
        self,
        record_time=False,
        error_wrt_each_grain=True,
        solver=None,
        verbose=True,
        backend="auto",
        **kwargs,
    ):
        """
        Find the set of weights W for which the APD generated by (X,As,W) is optimal with respect to target masses.
        """
        if self.Y is None:
            self.assemble_pixels()
        if error_wrt_each_grain:
            vol_tol = (self.error_tolerance) * self.target_masses
            if verbose:
                print("Solver tolerance is with respect to each grain separately.")
                print("Smallest tol: ", torch.min(vol_tol))
        else:
            vol_tol = self.error_tolerance * torch.min(self.target_masses)
            if verbose:
                print("Solver tolerance is with respect to the smallest grain.")
                print("Tolerance: ", vol_tol)

        defaultKwargs = {}
        if solver is None:
            solver = "bfgs"
            defaultKwargs = {
                "gtol": vol_tol,
                "xtol": 0,
                "disp": 1 if verbose else 0,
                "max_iter": 1000,
            }

        def fun(W):
            return self.OT_dual_function(W=W, backend=backend)

        # Solve the optimisation problem
        kwargs = {**defaultKwargs, **kwargs}

        start = time.time()
        res = minimize_torch(
            fun, self.W, method=solver, disp=1 if verbose else 0, options=kwargs
        )
        time_taken = time.time() - start
        if verbose:
            print("It took", time_taken, "seconds to find optimal W.")

        W = res.x

        self.W = W
        self.w = LazyTensor(self.W.view(self.N, 1, 1))
        self._W_from_heuristic = False  # optimised W must survive future set_pixels calls
        if record_time:
            self.time_to_find_W = time_taken

    def adjust_X(self, backend="auto"):
        """
        Move each seed to the mass-weighted centroid of its grain (Lloyd step).
        """
        if not self.optimality:
            print("Find optimal W first!")
        else:
            D_ij = ((self.y - self.x) | self.a.matvecmult(self.y - self.x)) - self.w
            grain_indices = D_ij.argmin(dim=0, backend=backend).ravel()
            normalisation = torch.bincount(grain_indices, self.PS, minlength=self.N)
            new_X0 = (
                torch.bincount(grain_indices, self.PS * self.Y[:, 0], minlength=self.N)
                / normalisation
            )
            new_X1 = (
                torch.bincount(grain_indices, self.PS * self.Y[:, 1], minlength=self.N)
                / normalisation
            )

            if self.D == 3:
                new_X2 = (
                    torch.bincount(
                        grain_indices, self.PS * self.Y[:, 2], minlength=self.N
                    )
                    / normalisation
                )
                self.X = torch.stack([new_X0, new_X1, new_X2], dim=1)
            else:
                self.X = torch.stack([new_X0, new_X1], dim=1)

            self.optimality = False
            self.x = LazyTensor(self.X.view(self.N, 1, self.D))

    def Lloyds_algorithm(self, K=5, verbosity_level=1, backend="auto"):
        for k in range(K):
            if verbosity_level > 0:
                print("Lloyds iteration:", k)

            verbose = True if verbosity_level == 2 else False
            self.find_optimal_W(verbose=verbose)
            self.check_optimality()
            self.adjust_X(backend=backend)

        # Final weight optimisation after the last seed-position update
        if verbosity_level > 0:
            print("Lloyds final optimisation")
        self.find_optimal_W(verbose=(verbosity_level == 2))
        self.check_optimality()
