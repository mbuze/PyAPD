import math
import time

import torch
from matplotlib import pyplot as plt
from numpy.polynomial import Legendre
from pykeops.torch import LazyTensor
from torchmin import minimize as minimize_torch

from .apds import apd_system
from .log_res_utils import (
    assemble_design_matrix,
    calculate_moments_from_data,
    convert_from_phys_to_lr,
    convert_theta_between_bases,
    gridify_Y_I,
    physical_heuristic_guess,
    reorder_variables,
)


class min_diagram_system:
    """
    A minimisation diagram system.
    """

    def __init__(
        self,
        Y=None,  # pixels / voxels
        grain_map=None,  # grain map
        pixel_params=None,
        theta=None,  # parameters
        ho=2,
        eps=1e-2,
        basis=Legendre,
        heuristic_guess=True,
        data_rescaling_type="centered_unit_interval",
        dt=torch.float64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # convenience constructor options:
        N=10,
        D=2,
        ani_thres=0.25,
        pixel_size_prefactor=2,
        seed=-1,
    ):
        """
        Construct a minimisation diagram system.

        Parameters
        ----------
        Y : Tensor, shape (M, D), optional
            Pixel / voxel positions. Generated from a fresh `apd_system` if not
            provided.
        I : Tensor, shape (M,), optional
            Grain-index map (integer labels 0..N-1). Required when Y is given.
        pixel_params : tuple of int, optional
            Pixel-grid resolution, e.g. ``(M, M)``. Inferred when Y is not
            provided.
        theta : Tensor, shape (N, K), optional
            Polynomial coefficient matrix. Initialised via heuristic or zeros
            if not provided.
        ho : int
            Highest polynomial order. Default: 2.
        eps : float
            Regularisation divisor applied to the design matrix. Default: 1e-2.
        basis : polynomial class
            Basis family from ``numpy.polynomial`` (Legendre, Polynomial, ...).
            Default: ``Legendre``.
        heuristic_guess : bool
            Initialise theta from the APD physical heuristic. Default: True.
        data_rescaling_type : str
            Pixel rescaling mode: ``"centered_unit_interval"`` (default),
            ``"unit_interval"``, or any other string for no rescaling.
        dt : torch.dtype
            Floating-point dtype. Default: ``torch.float64``.
        device : str
            Compute device. Default: CUDA if available, else CPU.
        N : int
            Number of grains when Y is not provided. Default: 10.
        D : int
            Spatial dimension when Y is not provided. Default: 2.
        ani_thres : float
            Anisotropy threshold for the default APD generator. Default: 0.25.
        pixel_size_prefactor : int
            Pixel-count multiplier for the default APD generator. Default: 2.
        seed : int
            Random seed for the default APD generator (< 0 means unset).
            Default: -1.
        """
        self.ho = ho
        self.eps = eps
        self.basis = basis
        self.data_rescaling_type = data_rescaling_type
        self.heuristic_guess = heuristic_guess

        self.N = N
        self.D = D
        self.ani_thres = ani_thres
        self.pixel_size_prefactor = pixel_size_prefactor

        self.dt = dt
        self.device = device
        # torch.set_default_dtype(dt)
        # torch.set_default_device(device)

        self.set_grain_map(Y, grain_map, pixel_params)
        self.set_theta(theta)
        self.assemble_design_matrix()

    def update_lr_data(
        self,
        # Y = None,
        ho=None,
        eps=None,
        basis=None,
        # data_rescaling_type = None,
    ):
        # Y = self.Y if Y is None else Y
        counter = 0
        if basis is not None and basis is not self.basis:
            self.theta = convert_theta_between_bases(
                self.theta,
                ho=self.ho,
                D=self.D,
                basis_end=basis,
                basis_start=self.basis,
            )
            self.basis = basis
            counter += 1
        if ho is not None and ho <= self.ho:
            print("ho not adjusted -- it only makes sense to increase ho!")
        if ho is not None and ho > self.ho:
            self.theta = reorder_variables(self.theta, self.D, self.ho, ho)
            self.ho = ho
            counter += 1

        # if data_rescaling_type is not None:
        #     self.data_rescaling_type = data_rescaling_type
        #     self.set_grain_map(Y=Y,I=self.I)
        #     #self.assemble_design_matrix()
        if eps is not None and eps is not self.eps:
            self.eps = eps
            counter += 1

        if counter > 0:
            self.assemble_design_matrix()

    def set_grain_map(self, Y=None, grain_map=None, pixel_params=None):
        """
        Set the pixel positions and grain-index map.

        Parameters
        ----------
        Y : Tensor, shape (M, D), optional
            Pixel positions. A random APD is generated if not provided.
        I : Tensor, shape (M,), optional
            Integer grain labels 0..N-1, one per pixel.
        pixel_params : tuple of int, optional
            Grid resolution (used for plotting). Inferred when Y is None.
        """
        if Y is None:
            apd = apd_system(
                N=self.N,
                D=self.D,
                pixel_size_prefactor=self.pixel_size_prefactor,
                ani_thres=self.ani_thres,
                dt=self.dt,
            )
            apd.assemble_pixels()
            Y = apd.Y
            grain_map = apd.assemble_apd()
            pixel_params = apd.pixel_params
        if self.data_rescaling_type == "centered_unit_interval":
            Y2 = Y - Y.mean((0), keepdim=True)
            Y2 /= Y2.max((0), keepdim=True).values
        elif self.data_rescaling_type == "unit_interval":
            Y2 = Y / Y.abs().max((0), keepdim=True).values
        else:
            Y2 = Y
        self.Y = Y2
        self.I = grain_map
        M, D = self.Y.shape
        self.N = (self.I.max() + 1).item()
        self.D = D
        self.M = M
        self.pixel_params = pixel_params

    def set_theta(self, theta=None):
        """
        Set the polynomial coefficient matrix theta.

        Parameters
        ----------
        theta : Tensor, shape (N, K), optional
            Coefficient matrix where K = C(D+ho, ho). If not provided, theta is
            initialised from the APD heuristic (when `heuristic_guess` is True
            and ho <= 2) or from zeros.
        """
        if theta is None:
            self.K = math.comb(self.D + self.ho, self.ho)
            if self.heuristic_guess:
                if self.ho < 3:
                    phys_guess = physical_heuristic_guess(self.I, self.Y, ho=self.ho)
                    guess = convert_from_phys_to_lr(phys_guess, ho=self.ho)
                    guess = guess - guess[0, :]
                    self.theta = convert_theta_between_bases(
                        guess, ho=self.ho, D=self.D, basis_end=self.basis
                    ).to(self.device)
                else:
                    phys_guess = physical_heuristic_guess(self.I, self.Y, ho=2)
                    guess = convert_from_phys_to_lr(phys_guess, ho=2)
                    guess = guess - guess[0, :]
                    guess = reorder_variables(guess, self.D, 2, self.ho)
                    self.theta = convert_theta_between_bases(
                        guess, ho=self.ho, D=self.D, basis_end=self.basis
                    ).to(self.device)
            else:
                self.theta = torch.zeros((self.N, self.K)).to(self.device)
        else:
            # TODO: add assertion check
            self.theta = theta.to(self.device)

        self.theta_l = LazyTensor(self.theta.view(self.N, 1, self.K))

    def assemble_design_matrix(self):
        self.design_matrix = assemble_design_matrix(
            self.Y, ho=self.ho, basis=self.basis, eps=self.eps
        ).to(self.device)
        self.K = self.design_matrix.shape[1]
        self.dml = LazyTensor(self.design_matrix.view(1, self.M, self.K))

    def objective_function(self, theta, backend="auto"):
        self.theta = theta
        self.theta_l = LazyTensor(self.theta.view(self.N, 1, self.K))
        first_sum = torch.sum(
            torch.index_select(self.theta, 0, self.I) * self.design_matrix
        )
        second_sum = (
            (self.theta_l * self.dml)
            .sum(dim=2, backend=backend)
            .logsumexp(dim=0, backend=backend)
            .sum(dim=0)
        )
        return -(1 / self.M) * (first_sum - second_sum)

    def fit_theta(
        self, record_time=False, solver=None, verbose=True, backend="auto", **kwargs
    ):
        """
        Fit theta by maximising the log-likelihood objective via L-BFGS.

        Parameters
        ----------
        record_time : bool
            Store elapsed time in ``self.fit_time``. Default: False.
        solver : str, optional
            Solver name passed to ``torchmin.minimize``. Defaults to
            ``'l-bfgs'``.
        verbose : bool
            Print solver progress and elapsed time. Default: True.
        backend : str
            KeOps backend (``"auto"``, ``"CPU"``, ``"GPU"``). Default: ``"auto"``.
        **kwargs
            Additional options forwarded to the solver.
        """
        defaultKwargs = {}
        if solver is None:
            solver = "l-bfgs"
            defaultKwargs = {
                "gtol": 1e-8,
                "xtol": -1e-10,
                "disp": 2 if verbose else 0,
                "max_iter": 10,
            }
        theta_t = torch.zeros(self.theta.shape)

        def fun(theta_red):
            theta_t[1:, :] = theta_red
            return self.objective_function(theta_t, backend=backend)

        # Solve the optimisation problem
        kwargs = {**defaultKwargs, **kwargs}

        start = time.time()
        minimize_torch(fun, self.theta[1:, :], method=solver, options=kwargs)
        time_taken = time.time() - start
        if verbose:
            print("It took", time_taken, "seconds to fit theta.")

    def assemble_diagram(self):
        """
        Assign each pixel to the grain whose polynomial function is minimal.

        Returns
        -------
        Tensor, shape (M,)
            Integer grain indices 0..N-1, one per pixel.
        """
        self.theta_l = LazyTensor(self.theta.view(self.N, 1, self.K))
        return (
            (-self.theta_l * self.dml).sum(dim=2).argmin(dim=0).ravel().to(self.device)
        )

    def plot_diagram(self, color_by=None):
        maxes = self.Y.max((0), keepdims=True).values[0]
        mins = self.Y.min((0), keepdims=True).values[0]
        dom_x = [mins[0], maxes[0]]
        dom_y = [mins[1], maxes[1]]
        domain = torch.tensor([dom_x, dom_y])
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10.5, 10.5, forward=True)
        I_new = gridify_Y_I(
            self.Y.cpu(),
            self.I.cpu(),
            domain.cpu(),
            self.pixel_params,
            color_by=color_by,
        )
        ax[0].imshow(I_new, origin="lower")
        diagram = self.assemble_diagram()
        I_new = gridify_Y_I(
            self.Y.cpu(),
            diagram.cpu(),
            domain.cpu(),
            self.pixel_params,
            color_by=color_by,
        )
        ax[1].imshow(I_new, origin="lower")
        return fig, ax

    def apd_from_grain_map(self):
        guess = physical_heuristic_guess(self.I, self.Y, ho=2)
        moments = calculate_moments_from_data(self.I, self.Y, ho=2)

        As_guess1 = torch.stack((guess[:, 0], guess[:, 1]), dim=1)
        As_guess2 = torch.stack((guess[:, 1], guess[:, 2]), dim=1)
        As_guess = torch.stack((As_guess1, As_guess2), dim=2)

        # dets =  torch.sqrt(torch.linalg.det(As_guess))
        # As_guess1 = torch.stack((guess[:,0]/dets,guess[:,1]/dets),dim=1)
        # As_guess2 = torch.stack((guess[:,1]/dets,guess[:,2]/dets),dim=1)
        # As_guess = torch.stack((As_guess1,As_guess2),dim=2)

        XX = guess[:, 3:5]
        maxes = self.Y.max((0), keepdims=True).values[0]
        mins = self.Y.min((0), keepdims=True).values[0]
        dom_x = [mins[0], maxes[0]]
        dom_y = [mins[1], maxes[1]]
        domain = torch.tensor([dom_x, dom_y])
        return apd_system(
            X=XX.contiguous(),
            domain=domain,
            pixel_params=self.pixel_params,
            As=As_guess.contiguous(),
            target_masses=(moments[:, 0] / len(self.Y)).contiguous(),
            heuristic_W=True,
            dt=self.dt,
        )
