"""
MTEX interface for PyAPD via matlab.engine.

Usage
-----
Set the environment variable MTEX_PATH to your MTEX installation, e.g.:

    export MTEX_PATH=~/MEGA/academic_work/projects/min_diagrams/code/mtex-6.1.0

Then use as a context manager:

    from PyAPD.mtex_interface import MTEXEngine

    with MTEXEngine() as eng:
        # eng.eng is the raw matlab.engine handle
        # MTEX is already on the MATLAB path
        eng.eval("some_mtex_function()")

Or manage lifecycle manually:

    eng = MTEXEngine()
    eng.start()
    eng.eval("some_mtex_function()")
    eng.stop()

Grain finding + export
----------------------

    with MTEXEngine() as eng:
        info = eng.find_grains("path/to/scan.ang", misorientation_threshold=1.0, min_pixels=10)
        paths = eng.export_grain_metadata("/tmp/output/")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matlab.engine

_DEFAULT_MTEX_PATH = os.path.expanduser(
    "~/MEGA/academic_work/projects/min_diagrams/code/mtex-6.1.0"
)


class MTEXEngine:
    """Thin wrapper around matlab.engine that loads MTEX on startup."""

    def __init__(self, mtex_path: str | None = None):
        """
        Parameters
        ----------
        mtex_path : str, optional
            Path to MTEX installation. Falls back to the MTEX_PATH environment
            variable, then to the default install location.
        """
        self.mtex_path = mtex_path or os.environ.get("MTEX_PATH") or _DEFAULT_MTEX_PATH
        if not os.path.isdir(self.mtex_path):
            raise FileNotFoundError(
                f"MTEX path not found: {self.mtex_path}\n"
                "Set the MTEX_PATH environment variable or pass mtex_path= explicitly."
            )
        self.eng: matlab.engine.MatlabEngine | None = None

    def start(self) -> MTEXEngine:
        """Start the MATLAB engine and initialise MTEX."""
        import matlab.engine  # lazy import; requires matlabengine package

        # On Linux, MATLAB may fail to load if libstdc++ is not the system
        # version.  Ensure LD_PRELOAD points to the system libstdc++ before
        # spawning the MATLAB process.  This mirrors running:
        #   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab
        _LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        if os.path.exists(_LD_PRELOAD):
            os.environ.setdefault("LD_PRELOAD", _LD_PRELOAD)

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(self.mtex_path), nargout=0)
        self.eng.startup(nargout=0)  # MTEX startup script
        return self

    def stop(self):
        """Quit the MATLAB engine."""
        if self.eng is not None:
            self.eng.quit()
            self.eng = None

    # Context manager support
    def __enter__(self) -> MTEXEngine:
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def eval(self, expr: str, **kwargs):
        """Thin pass-through to eng.eval.

        Defaults to ``nargout=0`` so that MATLAB assignment statements
        (e.g. ``ebsd = EBSD.load(...)``) are treated as statements rather
        than expressions, which avoids the "Incorrect use of '=' operator"
        SyntaxError raised by some MATLAB Engine versions.
        """
        if self.eng is None:
            raise RuntimeError("Engine not started. Use start() or a context manager.")
        kwargs.setdefault("nargout", 0)
        return self.eng.eval(expr, **kwargs)

    def find_grains(
        self,
        ebsd_path: str,
        misorientation_threshold: float = 1.0,
        min_pixels: int = 10,
    ) -> dict:
        """Load an EBSD file, compute grains, and filter small grains.

        Ports the logic from ``clean_example_version.m`` via sequential
        ``eng.eval()`` calls; no ``.m`` file is needed at runtime.

        After this method returns, the MATLAB workspace contains ``ebsd``,
        ``grains``, and ``ipfKey`` variables that are used by
        :meth:`export_grain_metadata`.

        Parameters
        ----------
        ebsd_path : str
            Absolute path to the EBSD file (e.g. ``.ang``, ``.ctf``).
            MTEX auto-detects the format.
        misorientation_threshold : float, optional
            Grain boundary misorientation threshold in degrees (default 1.0).
        min_pixels : int, optional
            Grains with this many pixels or fewer are removed before the
            second grain-finding pass (default 10).

        Returns
        -------
        dict
            ``{'n_grains': int, 'n_pixels': int}``
        """
        if self.eng is None:
            raise RuntimeError("Engine not started. Use start() or a context manager.")

        ebsd_path = str(Path(ebsd_path).resolve())

        # Load EBSD file (MTEX auto-detects format)
        self.eval(f"ebsd = EBSD.load('{ebsd_path}');")

        # Gridify for rectangular scans (no-op for non-rectangular)
        self.eval("ebsd = ebsd.gridify;")

        # First grain-finding pass
        self.eval(
            f"[grains, ebsd.grainId, ebsd.mis2mean] = "
            f"calcGrains(ebsd('indexed'), 'angle', {misorientation_threshold}*degree);"
        )

        # Remove small grains from ebsd
        self.eval(f"ebsd(grains(grains.numPixel <= {min_pixels})) = [];")

        # Second grain-finding pass on cleaned ebsd
        self.eval(
            f"[grains, ebsd.grainId, ebsd.mis2mean] = "
            f"calcGrains(ebsd('indexed'), 'angle', {misorientation_threshold}*degree);"
        )

        # Build IPF colour key (stored for use in export_grain_metadata)
        self.eval("ipfKey = ipfColorKey(ebsd);")

        n_grains = int(self.eng.eval("length(grains)", nargout=1))
        n_pixels = int(self.eng.eval("length(ebsd('indexed'))", nargout=1))

        return {"n_grains": n_grains, "n_pixels": n_pixels}

    def export_grain_metadata(self, output_dir: str) -> dict:
        """Write the three grain-metadata text files to ``output_dir``.

        Must be called after :meth:`find_grains` (requires ``ebsd``,
        ``grains``, and ``ipfKey`` in the MATLAB workspace).

        Writes:

        * ``grain_map.txt`` — x, y, grainId per indexed pixel
        * ``grains_mean_orientation.txt`` — phi1, Phi, phi2 per grain
        * ``grains_coloring.txt`` — IPF-Z RGB colour per grain

        Parameters
        ----------
        output_dir : str
            Directory to write the three files into (created if absent).

        Returns
        -------
        dict
            Mapping of ``{'grain_map', 'mean_orientation', 'coloring'}``
            to their absolute file paths.
        """
        if self.eng is None:
            raise RuntimeError("Engine not started. Use start() or a context manager.")

        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        grain_map_path = str(out / "grain_map.txt")
        orient_path = str(out / "grains_mean_orientation.txt")
        color_path = str(out / "grains_coloring.txt")

        # Pixel-level grain map: x, y, grainId
        self.eval(
            f"T = table(ebsd('indexed').x, ebsd('indexed').y, ebsd('indexed').grainId);"
            f"writetable(T, '{grain_map_path}', 'Delimiter', ' ');"
        )

        # Per-grain mean orientation (Euler angles in radians)
        self.eval(
            f"T = table(grains.meanOrientation.phi1, "
            f"grains.meanOrientation.Phi, grains.meanOrientation.phi2);"
            f"writetable(T, '{orient_path}', 'Delimiter', ' ');"
        )

        # Per-grain IPF-Z colour
        self.eval(
            f"grain_color_map = ipfKey.orientation2color(grains.meanOrientation);"
            f"T = table(grain_color_map);"
            f"writetable(T, '{color_path}', 'Delimiter', ' ');"
        )

        return {
            "grain_map": grain_map_path,
            "mean_orientation": orient_path,
            "coloring": color_path,
        }
