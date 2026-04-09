import numpy as np
import torch


def sample_seeds_with_exclusion(
    n, dim=3, radius_prefactor=1e-1, number_of_attempts=None, verbose=False
):
    """Sample n points from a uniform distribution on [0,1]^dim, but disregard points too close to each other.
    Used as seed points for the grains.
    Input arguments:

    n - number of seed points
    dim - dimension (optional, default is dim = 3)
    radius_prefactor - prefactor in front of the radius of exclusion which is given by radius_prefactor * n**(-1/dim) (optional, default is radius_prefactor =  0.1)
    number_of_attemps - how many times to try to generate a new seed point
    verbose - whether to print out how many proposed seed points have been rejected (optional, default is verbose = False)
    """
    if number_of_attempts is None:
        number_of_attempts = 100 * n

    radius = radius_prefactor * (n ** (-1 / dim))
    X = torch.rand(1, dim)
    counter = 0
    while len(X) < n:
        x_new = torch.rand(1, dim)
        if torch.min(torch.norm(X - x_new, dim=1)) > radius:
            X = torch.cat((X, x_new), dim=0)
        else:
            counter += 1
            if counter > number_of_attempts:
                print("Radius too large to have feasible chance of generating a sample")
                print("Only", len(X), "seed points have been generated.")
                break

    if verbose:
        print(counter, "proposed seed points have been excluded.")

    return X


def sample_spd_matrices_perturbed_from_identity(n, dim=3, amp=0.01):
    """Generate a collection of dim x dim spd matrices.
    Used to specify the anisotropy of each grain.

    Input arguments:

    n - number of matrices

    dim - dimension (optional, default is dim = 3).

    amp - amplitute of the perturbation from identity (optional, default is amp=0.01).
    """
    a = torch.stack((torch.eye(dim),) * n)
    a = a + amp * (2 * torch.rand(n, dim, dim) - 1)
    a = a @ a.transpose(-1, -2)
    return a


def convert_axes_and_angle_to_matrix_2D(a, b, theta):
    """Given the semi-major and semi-minor axis of an ellipse and an angle orientation, return the  2x2 positive semi-definite matrix associated with it.
    It can be used to generate 2D normalised anistropy matrices from 2D EBSD data.

    Input arguments:

    a - major axis (necessary)

    b - minor axis (necessary)

    theta - orientation angle (necessary)
    """
    a11 = (1 / a**2) * np.cos(theta) ** 2 + (1 / b**2) * np.sin(theta) ** 2
    a22 = (1 / a**2) * np.sin(theta) ** 2 + (1 / b**2) * np.cos(theta) ** 2
    a12 = ((1 / a**2) - (1 / b**2)) * np.cos(theta) * np.sin(theta)
    A = torch.tensor([[a11, a12], [a12, a22]])
    return A


def sample_normalised_spd_matrices(N, dim=3, ani_thres=0.5):
    """Generate a collection of n normalised random dim x dim symmetric positive definite matrices.
    Used to specify preferred orientation and aspect ratio of each grain, while retaining the normalisation constraint (determinant equal to 1).

    Input arguments:
    n - number of matrices
    dim - dimension (optional, default is dim = 3).
    ani_thres - acceptable level of anisotropy, values between [0,1], close to 1 means we accept any level of anisotropy, close to 0 that we want next no anisotropy (optional, default is 0.5). Note that setting this parameter 1 can make the resulting optimisation problem ill-conditioned.
    """
    if dim == 2:
        a_s = (
            torch.ones(N)
            if ani_thres == 0
            else torch.distributions.Uniform(1 - ani_thres, 1).sample([N])
        )
        b_s = 1.0 / a_s
        thetas = torch.distributions.Uniform(0, 2 * np.pi).sample([N])

        ss = torch.sin(thetas)
        cc = torch.cos(thetas)
        rots = torch.stack(
            [torch.stack([cc, -ss], dim=1), torch.stack([ss, cc], dim=1)], dim=2
        )
        IIs = torch.stack(
            [
                torch.stack([1 / a_s**2, torch.tensor([0.0] * N)], dim=1),
                torch.stack([torch.tensor([0.0] * N), 1 / b_s**2], dim=1),
            ],
            dim=2,
        )
        As = rots @ IIs @ torch.transpose(rots, 1, 2)
    else:
        a_s = (
            torch.ones(N)
            if ani_thres == 0
            else torch.distributions.Uniform(1 - ani_thres, 1).sample([N])
        )
        b_s = (
            torch.ones(N)
            if ani_thres == 0
            else torch.distributions.Uniform(
                1 - ani_thres, 1.0 / (1 - ani_thres)
            ).sample([N])
        )
        c_s = 1.0 / (a_s * b_s)
        alphas = torch.distributions.Uniform(0, 2 * np.pi).sample([N])
        betas = torch.distributions.Uniform(0, 2 * np.pi).sample([N])
        gammas = torch.distributions.Uniform(0, 2 * np.pi).sample([N])

        ss = torch.sin(alphas)
        cc = torch.cos(alphas)

        rots_x = torch.stack(
            [
                torch.stack([torch.ones(N), torch.zeros(N), torch.zeros(N)], dim=1),
                torch.stack([torch.zeros(N), cc, -ss], dim=1),
                torch.stack([torch.zeros(N), ss, cc], dim=1),
            ],
            dim=2,
        )
        ss = torch.sin(betas)
        cc = torch.cos(betas)

        rots_y = torch.stack(
            [
                torch.stack([cc, torch.zeros(N), ss], dim=1),
                torch.stack([torch.zeros(N), torch.ones(N), torch.zeros(N)], dim=1),
                torch.stack([-ss, torch.zeros(N), cc], dim=1),
            ],
            dim=2,
        )

        ss = torch.sin(gammas)
        cc = torch.cos(gammas)

        rots_z = torch.stack(
            [
                torch.stack([cc, -ss, torch.zeros(N)], dim=1),
                torch.stack([ss, cc, torch.zeros(N)], dim=1),
                torch.stack([torch.zeros(N), torch.zeros(N), torch.ones(N)], dim=1),
            ],
            dim=2,
        )

        rots = rots_z @ rots_y @ rots_x

        IIs = torch.stack(
            [
                torch.stack([1 / a_s**2, torch.zeros(N), torch.zeros(N)], dim=1),
                torch.stack([torch.zeros(N), 1 / b_s**2, torch.zeros(N)], dim=1),
                torch.stack([torch.zeros(N), torch.zeros(N), 1 / c_s**2], dim=1),
            ],
            dim=2,
        )

        As = rots @ IIs @ torch.transpose(rots, 1, 2)
    return As


def initial_guess_heuristic(As, TVs, D):
    """Compute the initial guess for the weights based on the heuristic from
     Kirubel Teferra & David J. Rowenhorst (2018)
     Direct parameter estimation for generalised balanced power diagrams,
     Philosophical Magazine Letters, 98:2, 79-87,
     DOI: 10.1080/09500839.2018.1472399

    Input arguments:
    As - set of anisotropy matrices (necessary)
    TVs - set of target volumes of grains (necessary)
    D - dimension of the problem (necessary)
    """
    dets = torch.linalg.det(As)
    prefactor = (1.0 / torch.pi) if D == 2 else 3.0 / (4.0 * torch.pi)
    return (prefactor * TVs * torch.sqrt(dets)) ** (2.0 / D)


def ellipse_ratio_angle(As):
    """Extract aspect ratio and orientation angle from 2D SPD matrices.

    Parameters
    ----------
    As : torch.Tensor, shape (N, 2, 2)
        Symmetric positive definite matrices.

    Returns
    -------
    torch.Tensor, shape (N, 2)
        Columns are ``[ratio, angle]`` where

        * ``ratio = sqrt(lambda_max / lambda_min)`` — always >= 1
        * ``angle = atan2(v1, v0)`` of the eigenvector for ``lambda_min``,
          normalised to ``[-pi/2, pi/2]``
    """
    eigvals, eigvecs = torch.linalg.eigh(As)  # ascending eigenvalues
    lambda_min = eigvals[:, 0]
    lambda_max = eigvals[:, 1]

    ratio = torch.sqrt(lambda_max / lambda_min)

    # Eigenvector for lambda_min (first column)
    v = eigvecs[:, :, 0]  # [N, 2]
    angle = torch.atan2(v[:, 1], v[:, 0])

    # Normalise to [-pi/2, pi/2]
    angle = torch.where(angle > torch.pi / 2, angle - torch.pi, angle)
    angle = torch.where(angle < -torch.pi / 2, angle + torch.pi, angle)

    return torch.stack([ratio, angle], dim=1)


def ellipse_A_from_ratio_angle(ratio_angle, det=1.0):
    """Reconstruct 2D SPD matrices from aspect ratio and orientation angle.

    Inverse of :func:`ellipse_ratio_angle`.

    Parameters
    ----------
    ratio_angle : torch.Tensor, shape (N, 2)
        Columns are ``[ratio, angle]`` with ratio > 0.
    det : float, optional
        Target determinant of the output matrices (default 1.0).

    Returns
    -------
    torch.Tensor, shape (N, 2, 2)
        SPD matrices with the requested determinant, aspect ratio, and
        orientation.
    """
    if ratio_angle.ndim != 2 or ratio_angle.size(1) != 2:
        raise ValueError("Input must be shape (N, 2) with columns [ratio, angle].")

    r = ratio_angle[:, 0]
    theta = ratio_angle[:, 1]

    if (r <= 0).any():
        raise ValueError("All ratios must be positive.")

    c = torch.cos(theta)
    s = torch.sin(theta)

    # Eigenvalues: lambda_min * lambda_max = det, lambda_max / lambda_min = r^2
    # => lambda_min = sqrt(det) / r,  lambda_max = r * sqrt(det)
    scale = ratio_angle.new_tensor(det).sqrt() / r
    lam_min = scale
    lam_max = (r**2) * scale

    A11 = lam_min * c**2 + lam_max * s**2
    A22 = lam_min * s**2 + lam_max * c**2
    A12 = (lam_min - lam_max) * c * s

    A = torch.stack(
        [torch.stack([A11, A12], dim=1), torch.stack([A12, A22], dim=1)], dim=1
    )
    return A


def fit_grain_statistics_kde(apd_system, bandwidth="silverman"):
    """Fit a 3D KDE to grain statistics extracted from a fitted APD.

    Extracts ``[grain_area, ellipse_ratio, ellipse_angle]`` from
    ``apd_system`` and fits a kernel density estimate using
    ``openturns.KernelSmoothing`` with a Uniform kernel.

    Parameters
    ----------
    apd_system : apd_system
        A fitted :class:`apd_system` instance with ``As`` and
        ``target_masses`` attributes (2D, ``As`` shape (N, 2, 2)).
    bandwidth : str or array-like, optional
        ``'silverman'`` (default) computes bandwidth via Silverman's rule.
        Otherwise pass a length-3 array of bandwidths.

    Returns
    -------
    fittedDist : openturns.Distribution
        The fitted KDE distribution (3D).
    raw_data : np.ndarray, shape (N, 3)
        The data used to fit the KDE, columns
        ``[log_area, ellipse_ratio, ellipse_angle]`` where ``log_area``
        is the natural log of the normalised grain area fraction.
        Use ``np.exp(raw_data[:, 0])`` to recover normalised areas.

    Notes
    -----
    Requires ``openturns``.  Install with ``pip install openturns`` or
    ``pip install 'PyAPD[stats]'``.
    """
    try:
        import openturns as ot
    except ImportError as e:
        raise ImportError(
            "openturns is required for KDE fitting. "
            "Install it with: pip install openturns"
        ) from e

    ra = ellipse_ratio_angle(apd_system.As)  # (N, 2): ratio, angle
    masses = apd_system.target_masses
    areas_normalised = masses / masses.sum()  # fractions summing to 1
    log_areas = torch.log(areas_normalised).unsqueeze(1)  # (N, 1)
    data = torch.cat([log_areas, ra], dim=1)  # (N, 3): log_area, ratio, angle
    raw_data = data.cpu().numpy()

    sample = ot.Sample(
        [
            [float(raw_data[i, 0]), float(raw_data[i, 1]), float(raw_data[i, 2])]
            for i in range(raw_data.shape[0])
        ]
    )

    kernel = ot.Uniform()
    ks = ot.KernelSmoothing(kernel)

    if bandwidth == "silverman":
        bw = ot.KernelSmoothing().computeSilvermanBandwidth(sample)
    else:
        bw = ot.Point(list(bandwidth))

    fittedDist = ks.build(sample, bw)

    return fittedDist, raw_data


def sample_synthetic_grains(fittedDist, N_sim, raw_data, det=1.0, min_frac=0.5):
    """Sample synthetic grain parameters from a fitted KDE distribution.

    Parameters
    ----------
    fittedDist : openturns.Distribution
        Fitted distribution returned by :func:`fit_grain_statistics_kde`.
    N_sim : int
        Number of grains to sample.
    raw_data : np.ndarray, shape (N, 3)
        The array returned by :func:`fit_grain_statistics_kde`, with columns
        ``[log_area, ellipse_ratio, ellipse_angle]``.  Used to determine the
        smallest observed grain area for the floor.
    det : float, optional
        Target determinant for the reconstructed SPD matrices (default 1.0).
    min_frac : float, optional
        Sampled areas are clamped to >= ``min_frac * smallest_observed_area``.
        Default 0.5 (grains must be at least half the size of the smallest
        real grain).

    Returns
    -------
    target_masses : torch.Tensor, shape (N_sim,)
        Normalised grain areas (sum to 1, all positive).
    As : torch.Tensor, shape (N_sim, 2, 2)
        SPD matrices reconstructed from sampled (ratio, angle) pairs.
    """
    import numpy as np

    sim = np.array(fittedDist.getSample(N_sim))  # (N_sim, 3): log_area, ratio, angle

    # Exponentiate to recover normalised areas
    areas = np.exp(sim[:, 0])

    # Floor: min_frac × smallest area seen in the real data
    min_observed = np.exp(raw_data[:, 0].min())
    areas = np.clip(areas, min_frac * min_observed, None)
    areas = areas / areas.sum()
    target_masses = torch.from_numpy(areas).float()

    # Reconstruct SPD matrices; clamp ratio to be >= 1 (it should be by construction)
    ratio_angle = torch.from_numpy(sim[:, 1:3]).float()
    ratio_angle[:, 0] = ratio_angle[:, 0].clamp(min=1.0 + 1e-6)

    As = ellipse_A_from_ratio_angle(ratio_angle, det=det)

    return target_masses, As


def specify_volumes(
    n, crystal_types=1, ratios=None, volume_ratios=None, max_volume_deviation=10
):
    """Specify the target volumes of the grains. Grains can be grouped into T types (default T = 1)

    Input arguments:

    n - number of grains (necessary)

    crystal_types - number of grain types (optional, default is crystal_types = 1)
    ratios - approximate percentage number of grains of each type, e.g. ratios = [0.2, 0.5, 0.3] for 3 graintypes
    (optional, default is a random vector sampled from uniform distribution over the open standard (crystal_types − 1)-simplex.

    volume_ratios - e.g. volume_ratios = [1,4,5] for 3 grain types to indicate that grains of type 2 should be 4-times the size of grains of type 1
    and grains of type 3 should be 5 times the size of grains of type 1
    (optional, default is a random vector sampled from a Categorical distribution between 1 and max_volume_deviation

    max_volume_deviation - explained above (optional, default max_volume_deviation = 10)
    """
    if ratios is None:
        probs = torch.ones(crystal_types).to("cpu")
        c = torch.distributions.Dirichlet(probs)
        ratios = c.sample().to("cpu")

    rand = torch.rand(n).to("cpu")
    crystal = torch.ones(n).to("cpu")
    for k in range(crystal_types):
        crystal[sum(ratios[: (k + 1)]) <= rand] = k + 1

    if volume_ratios is None:
        probs = torch.tensor([[1] * max_volume_deviation] * crystal_types).to("cpu")
        c = torch.distributions.Categorical(probs)
        volume_ratios = ratios = c.sample().to("cpu") + 1

    volumes = torch.ones(n).to("cpu")
    for k in range(crystal_types):
        volumes[crystal == k + 1] = volume_ratios[k]

    return volumes / volumes.sum()
