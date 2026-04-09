import numpy as np
import torch

# from .geometry_utils import *


def load_setup_from_EBSD_data_2D(
    file="../../data/2D_basic_example/EBSD_example_2D_data.txt",
    seed_id=(1, 2),
    volumes_id=(3),
    orientation_id=(4, 5, 6),
    normalise_matrices=True,
    angle_in_degrees=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dt=torch.float32,
):
    """
    Load geometric setup from 2D EBSD data. The data is located in `/data`.
    The file has to be such that each line corresponds to one grain.
    The default settings are for the deafult file where each line looks as follows:

    1 11.546 19.120 8.426e+002 21.38 13.78 103.5 973

    where
    1 is grain label
    (11.546,19.120) is the centroid of the grain (2D)
    8.426e+002 is the volume of the grain
    (21.38, 13.78, 103.5) defines a 2D ellipse (major axis, minor axis, rotation angle)
    973 is (presumably) the number of pixels (which determines the volume)

    Input:
    file - location of the file, should be of the form "../data/..." to load some data from the data folder
    (optional, default is "../data/2D_basic_example/EBSD_example_2D_data.txt" )
    seed_id - column ids storing location of each seed,
    volumes_id - column id storing volume of each seed,
    orientation_id - column ids storing orientation info of the grain (major axis, minor axis, rotation angle)
    normalise_matrices - whether to ensure that the anisotropy matrices have determinant equal to 1 or not (optional, default is True)
    device - "cuda" or "cpu" (optional default is "cuda" if it is available and "cpu" otherwise)
    dt - data type used by Torch (optional default is torch.float32)
    """
    # EBSD = np.loadtxt("../data/EBSD_data.txt")
    EBSD = torch.from_numpy(np.loadtxt(file)).to(device=device, dtype=dt)
    X = EBSD[:, seed_id]
    TV = EBSD[:, volumes_id]
    if normalise_matrices:
        ratios = EBSD[:, orientation_id[0]] / EBSD[:, orientation_id[1]]
        major_axes = torch.sqrt(ratios)
        minor_axes = 1 / major_axes
        thetas = (
            EBSD[:, orientation_id[2]] * np.pi / 180
            if angle_in_degrees
            else EBSD[:, orientation_id[2]]
        )
    else:
        major_axes = EBSD[:, orientation_id[0]]
        minor_axes = EBSD[:, orientation_id[1]]
        thetas = (
            EBSD[:, orientation_id[2]] * np.pi / 180
            if angle_in_degrees
            else EBSD[:, orientation_id[2]]
        )

    a11 = (1 / major_axes**2) * torch.cos(thetas) ** 2 + (
        1 / minor_axes**2
    ) * torch.sin(thetas) ** 2
    a22 = (1 / major_axes**2) * torch.sin(thetas) ** 2 + (
        1 / minor_axes**2
    ) * torch.cos(thetas) ** 2
    a12 = (
        ((1 / major_axes**2) - (1 / minor_axes**2))
        * torch.cos(thetas)
        * torch.sin(thetas)
    )
    a1 = torch.stack([a11, a12], dim=1)
    a2 = torch.stack([a12, a22], dim=1)
    A = torch.stack([a1, a2], dim=2)
    return X.contiguous(), A.contiguous(), TV.contiguous(), EBSD


# Maps TSL/OIM integer symmetry codes to MTEX-compatible Laue group strings.
# MTEX's loadEBSD_ang passes the # Symmetry token directly to crystalSymmetry();
# numeric TSL codes like 43 are not recognised by crystalSymmetry, so we convert
# them here to the standard Hermann-Mauguin names that MTEX does accept.
_TSL_TO_LAUE = {
    1: "1",       # triclinic
    2: "2/m",     # monoclinic
    20: "2/m",    # monoclinic (alternative TSL code)
    22: "mmm",    # orthorhombic
    3: "-3",      # trigonal
    32: "-3m",    # trigonal
    4: "4/m",     # tetragonal
    42: "4/mmm",  # tetragonal
    6: "6/m",     # hexagonal
    62: "6/mmm",  # hexagonal
    23: "m-3",    # cubic
    43: "m-3m",   # cubic (most common — BCC/FCC iron)
}


def write_ang_file(
    path,
    Y,
    euler_angles,
    grain_ids,
    pixel_size,
    ci=None,
    iq=None,
    phase_info=None,
):
    """Write a TSL/OIM-compatible ``.ang`` file from APD output.

    Parameters
    ----------
    path : str
        Output file path.
    Y : array-like, shape (M, 2)
        Pixel coordinates (x, y) in microns.
    euler_angles : array-like, shape (N, 3)
        Euler angles (phi1, Phi, phi2) in radians, one row per grain.
    grain_ids : array-like, shape (M,)
        Zero-based grain index for each pixel.  Use -1 for unindexed pixels.
    pixel_size : float
        Step size in microns (assumed equal in x and y).
    ci : array-like, shape (M,), optional
        Confidence index per pixel.  Defaults to 0.
    iq : array-like, shape (M,), optional
        Image quality per pixel.  Defaults to 0.
    phase_info : list of dict, optional
        Crystal phase metadata, one dict per phase (in 1-indexed order).
        Each dict may contain:
        ``'name'`` (str), ``'formula'`` (str),
        ``'symmetry'`` (int, TSL symmetry code — e.g. 43 for cubic m-3m),
        ``'lattice'`` (list of 6 floats: a b c alpha beta gamma in Å/°).
        TSL integer symmetry codes are automatically converted to
        MTEX-compatible Laue group strings (e.g. 43 → 'm-3m').

    Notes
    -----
    The function always writes a **complete rectangular grid**: grid positions
    that are not present in ``Y`` (e.g. unindexed pixels filtered before
    calling this function) are padded as unindexed entries
    (``phi1=phi2=Phi=4``, ``CI=-1``, ``phase=0``).  This is required because
    MTEX's ``loadEBSD_ang`` checks that the number of data lines equals
    ``NCOLS_ODD * NROWS`` for ``SqrGrid`` files.

    The data body follows the standard 10-column ANG layout:
    ``phi1 Phi phi2 x y IQ CI PhaseID DetectorIntensity Fit``
    """
    Y = np.asarray(Y, dtype=np.float64)
    euler_angles = np.asarray(euler_angles, dtype=np.float64)
    grain_ids = np.asarray(grain_ids, dtype=np.int32)
    M = Y.shape[0]

    if ci is None:
        ci = np.zeros(M, dtype=np.float64)
    else:
        ci = np.asarray(ci, dtype=np.float64)

    if iq is None:
        iq = np.zeros(M, dtype=np.float64)
    else:
        iq = np.asarray(iq, dtype=np.float64)

    # Build a (ix, iy) → row index lookup for fast grid traversal.
    # Round each coordinate to the nearest grid integer to avoid floating-point
    # mismatches (raw coordinates from h5oina can differ by ~1e-6 from exact multiples).
    ix_arr = np.round(Y[:, 0] / pixel_size).astype(np.int64)
    iy_arr = np.round(Y[:, 1] / pixel_size).astype(np.int64)
    coord_to_idx = {(int(ix_arr[j]), int(iy_arr[j])): j for j in range(M)}

    # Grid extent in integer grid coordinates
    ix_min, ix_max = int(ix_arr.min()), int(ix_arr.max())
    iy_min, iy_max = int(iy_arr.min()), int(iy_arr.max())
    ncols = ix_max - ix_min + 1
    nrows = iy_max - iy_min + 1

    with open(path, "w", newline="\n") as f:
        # Phase metadata — MTEX requires MTEX-compatible Laue group strings,
        # not TSL integer codes.  Convert via _TSL_TO_LAUE.
        if phase_info is not None:
            f.write("# TEM_PIXperUM          1.000000\n")
            f.write("#\n")
            for i, ph in enumerate(phase_info, start=1):
                raw_sym = ph.get("symmetry", 43)
                laue_str = _TSL_TO_LAUE.get(int(raw_sym), str(raw_sym))
                f.write(f"# Phase {i}\n")
                f.write(f"# MaterialName  \t{ph.get('name', 'Unknown')}\n")
                f.write(f"# Formula       \t{ph.get('formula', '')}\n")
                f.write("# Info          \t\n")
                f.write(f"# Symmetry              {laue_str}\n")
                lc = ph.get("lattice", [2.870, 2.870, 2.870, 90.0, 90.0, 90.0])
                f.write(
                    f"# LatticeConstants      "
                    f"{lc[0]:.3f} {lc[1]:.3f} {lc[2]:.3f}  "
                    f"{lc[3]:.3f}  {lc[4]:.3f}  {lc[5]:.3f}\n"
                )
                f.write("# NumberFamilies        0\n")
                f.write("#\n")

        f.write("# GRID: SqrGrid\n")
        f.write(f"# XSTEP: {pixel_size:.6f}\n")
        f.write(f"# YSTEP: {pixel_size:.6f}\n")
        f.write(f"# NCOLS_ODD: {ncols:d}\n")
        f.write(f"# NCOLS_EVEN: {ncols:d}\n")
        f.write(f"# NROWS: {nrows:d}\n")
        f.write("#\n")
        f.write("# Columns: phi1 Phi phi2 x y IQ CI PhaseID DetectorIntensity Fit\n")

        # Write the complete rectangular grid, row by row (y slow, x fast).
        # Grid positions absent from the input data are written as unindexed.
        for iy in range(iy_min, iy_max + 1):
            for ix in range(ix_min, ix_max + 1):
                x_coord = ix * pixel_size
                y_coord = iy * pixel_size
                j = coord_to_idx.get((ix, iy))
                if j is None:
                    # Position not in input data — pad as unindexed
                    f.write(
                        f"4.00000000 4.00000000 4.00000000 "
                        f"{x_coord:.6f} {y_coord:.6f} "
                        f"0.000 -1.000 0 0 180.000\n"
                    )
                else:
                    gid = int(grain_ids[j])
                    if gid == -1:
                        f.write(
                            f"4.00000000 4.00000000 4.00000000 "
                            f"{x_coord:.6f} {y_coord:.6f} "
                            f"0.000 -1.000 0 0 180.000\n"
                        )
                    else:
                        phi1, Phi, phi2 = euler_angles[gid]
                        f.write(
                            f"{phi1:.8f} {Phi:.8f} {phi2:.8f} "
                            f"{x_coord:.6f} {y_coord:.6f} "
                            f"{iq[j]:.3f} {ci[j]:.3f} 1 0 0.000\n"
                        )


def load_setup_from_h5oina(path, phase_id=None):
    """Load EBSD data from an Oxford Instruments ``.h5oina`` file.

    Parameters
    ----------
    path : str
        Path to the ``.h5oina`` file.
    phase_id : int, optional
        If given, keep only pixels whose phase matches this value.
        By default all indexed pixels (phase != 0) are returned.

    Returns
    -------
    dict with keys:

    ``'Y'`` : torch.Tensor, shape (M, 2)
        Pixel coordinates (x, y) as a float32 tensor, normalised to
        the bounding box so that all coordinates lie in ``[-1, 1]^2``.
        Use with :meth:`apd_system.set_pixels` for arbitrary-domain
        reconstructions.
    ``'PS'`` : torch.Tensor, shape (M,)
        Per-pixel size (area), uniform and equal to ``xstep * ystep``.
    ``'euler'`` : np.ndarray, shape (M, 3)
        Euler angles (phi1, Phi, phi2) in radians for each kept pixel.
    ``'phase_ids'`` : np.ndarray, shape (M,)
        Phase index (integer) for each kept pixel.
    ``'grain_ids'`` : None
        Placeholder; grain IDs require a separate grain-finding step
        (e.g. via :class:`MTEXEngine`).
    ``'xstep'`` : float
        x step size in the original file units (typically microns).
    ``'ystep'`` : float
        y step size in the original file units.
    ``'x_raw'`` : np.ndarray, shape (M,)
        Raw x coordinates (before normalisation).
    ``'y_raw'`` : np.ndarray, shape (M,)
        Raw y coordinates (before normalisation).

    Notes
    -----
    Requires ``h5py``.  For non-rectangular scans the returned ``Y``
    contains all valid pixels; pass directly to
    ``apd_system.set_pixels(Y, PS)``.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required to read .h5oina files. Install it with: pip install h5py"
        ) from e

    with h5py.File(path, "r") as f:
        xstep = float(f["1/EBSD/Header/X Step"][0])
        ystep = float(f["1/EBSD/Header/Y Step"][0])

        euler = f["1/EBSD/Data/Euler"][:].astype(np.float64)  # (N, 3)
        phase = f["1/EBSD/Data/Phase"][:].astype(np.int32)  # (N,)
        x = f["1/EBSD/Data/X"][:].astype(np.float64)
        y = f["1/EBSD/Data/Y"][:].astype(np.float64)

    # Euler angles: convert from degrees if necessary
    if np.nanmax(euler) > 2 * np.pi + 1e-3:
        euler = np.deg2rad(euler)

    # Build mask for valid (indexed) pixels
    if phase_id is not None:
        mask = phase == phase_id
    else:
        mask = phase != 0  # phase==0 is the standard notIndexed marker

    x = x[mask]
    y = y[mask]
    euler = euler[mask]
    phase = phase[mask]

    # Normalise coordinates to [-1, 1]^2 using the larger extent for both axes
    # so aspect ratio is preserved
    scale = max(x.max() - x.min(), y.max() - y.min())
    if scale == 0:
        scale = 1.0
    x_norm = 2.0 * (x - x.min()) / scale - 1.0
    y_norm = 2.0 * (y - y.min()) / scale - 1.0

    Y = torch.from_numpy(np.stack([x_norm, y_norm], axis=1)).to(torch.float32)
    ps_val = float(xstep * ystep)
    PS = torch.full((Y.shape[0],), ps_val, dtype=torch.float32)

    return {
        "Y": Y,
        "PS": PS,
        "euler": euler,
        "phase_ids": phase,
        "grain_ids": None,
        "xstep": xstep,
        "ystep": ystep,
        "x_raw": x,
        "y_raw": y,
    }


def get_rectangular_patch(Y, PS, grain_ids, x_range=None, y_range=None):
    """Crop an arbitrary pixel cloud to a rectangular window.

    Parameters
    ----------
    Y : array-like or torch.Tensor, shape (M, 2)
        Pixel coordinates.
    PS : array-like or torch.Tensor, shape (M,)
        Per-pixel sizes.
    grain_ids : array-like or torch.Tensor, shape (M,), or None
        Grain IDs per pixel (may be None).
    x_range : tuple (x_min, x_max) or None
        Inclusive x bounds.  ``None`` means use the full x extent.
    y_range : tuple (y_min, y_max) or None
        Inclusive y bounds.  ``None`` means use the full y extent.

    Returns
    -------
    Y_crop : torch.Tensor, shape (K, 2)
    PS_crop : torch.Tensor, shape (K,)
    grain_ids_crop : torch.Tensor or None, shape (K,)
    """
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(Y)
    if not isinstance(PS, torch.Tensor):
        PS = torch.as_tensor(PS)

    x = Y[:, 0]
    y = Y[:, 1]

    x_min = x_range[0] if x_range is not None else x.min().item()
    x_max = x_range[1] if x_range is not None else x.max().item()
    y_min = y_range[0] if y_range is not None else y.min().item()
    y_max = y_range[1] if y_range is not None else y.max().item()

    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    Y_crop = Y[mask]
    PS_crop = PS[mask]

    if grain_ids is not None:
        if not isinstance(grain_ids, torch.Tensor):
            grain_ids = torch.as_tensor(grain_ids)
        grain_ids_crop = grain_ids[mask]
    else:
        grain_ids_crop = None

    return Y_crop, PS_crop, grain_ids_crop
