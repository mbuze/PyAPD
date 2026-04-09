"""Smoke tests for PyAPD.mtex_interface.

Skipped automatically in CI (MATLAB not on PATH).
Run locally to verify the MTEX interface still works after changes.
"""

import shutil

import pytest

from PyAPD.mtex_interface import MTEXEngine

requires_matlab = pytest.mark.skipif(
    shutil.which("matlab") is None,
    reason="MATLAB not on PATH",
)


def test_bad_path_raises():
    with pytest.raises(FileNotFoundError, match="MTEX path not found"):
        MTEXEngine(mtex_path="/nonexistent/path")


@requires_matlab
def test_engine_starts_and_mtex_loads():
    """Start engine, load MTEX, run a trivial eval. (~30s)"""
    with MTEXEngine() as eng:
        result = eng.eval("1 + 1", nargout=1)
        assert result == 2.0
