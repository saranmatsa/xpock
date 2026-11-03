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

Note: the CLI smoke tests invoke :func:`xpock.cli.main` in-process
rather than in a subprocess. On this 6 GB RAM development machine a
subprocess re-import of scipy intermittently fails with
"DLL load failed: the paging file is too small" — an environment
limitation, not a code defect (CI runs on properly provisioned runners).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging


def _cli_main():
    from xpock.cli import main
    return main


def test_xpock_imports() -> None:
    """The installed package imports and exposes the public API."""
    import xpock

    assert xpock.__version__ == "0.1.0"

    from xpock.core import WaveEvent, WaveField  # noqa: F401
    from xpock.analysis import WaveAnalyzer  # noqa: F401
    from xpock.properties import PolarizationTexture  # noqa: F401


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Run xpock.cli.main in-process, capturing stdout and exit code."""
    buf = io.StringIO()
    code: int | None = None
    main = _cli_main()
    with contextlib.redirect_stdout(buf):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 1
    return code, buf.getvalue()


def test_cli_help_runs() -> None:
    """`xpock --help` exits 0 (CLI entry point loads)."""
    code, out = _run_cli(["--help"])
    assert code == 0, out
    assert "xpock" in out.lower() or "wave-structure" in out.lower()


def test_cli_config_smoke(tmp_path: Path) -> None:
    """`xpock config` writes a usable YAML config file."""
    out = tmp_path / "config_smoke.yaml"
    code, _ = _run_cli(["config", "--output", str(out)])
    assert code == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "grid:" in text and "temporal:" in text
