"""
Packaging reproducibility tests.

These validate the canonical installation path (root pyproject.toml,
package xpock from xpock/src):

1. xpock imports after `pip install -e .` from the repository root.
2. The `xpock` CLI entry point works.
3. A clean checkout installs without relying on the developer's
   existing environment (verified in CI: .github/workflows/ci.yml).

They are marked with pytest.mark.packaging so the local (uninstalled)
test run can skip them: run with

    pytest xpock/tests -m "not packaging"

when xpock is not installed, and the full suite after installation.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.packaging


def test_xpock_imports() -> None:
    """The installed package imports and exposes the public API."""
    import xpock

    assert xpock.__version__ == "0.1.0"

    from xpock.cli import main  # noqa: F401  CLI entry point exists
    from xpock.core import WaveEvent, WaveField  # noqa: F401
    from xpock.analysis import WaveAnalyzer  # noqa: F401
    from xpock.properties import PolarizationTexture  # noqa: F401


def test_cli_help_runs(tmp_path) -> None:
    """`python -m xpock.cli --help` exits 0 (CLI entry point loads)."""
    result = subprocess.run(
        [sys.executable, "-m", "xpock.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "xpock" in result.stdout.lower() or "wave-structure" in result.stdout.lower()


def test_cli_config_smoke(tmp_path) -> None:
    """`xpock config` writes a usable YAML config file."""
    out = tmp_path / "config_smoke.yaml"
    result = subprocess.run(
        [sys.executable, "-m", "xpock.cli", "config", "--output", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "grid:" in text and "temporal:" in text
