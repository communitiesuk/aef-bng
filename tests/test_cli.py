"""Tests for aef_bng.cli."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from aef_bng.cli import main


@pytest.mark.unit
class TestCli:
    """Tests for the Click CLI entry point."""

    def test_main_help(self) -> None:
        """--help prints usage and exits 0."""
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "AEF embeddings" in result.output

    def test_spark_run_help(self) -> None:
        """spark-run --help lists all options."""
        result = CliRunner().invoke(main, ["spark-run", "--help"])
        assert result.exit_code == 0
        assert "--bounds" in result.output
        assert "--years" in result.output
        assert "--table-name" in result.output

    def test_spark_run_requires_options(self) -> None:
        """spark-run without required options exits with a non-zero code."""
        result = CliRunner().invoke(main, ["spark-run"])
        assert result.exit_code != 0

    def test_verbose_flag_accepted(self) -> None:
        """--verbose flag is accepted without error."""
        result = CliRunner().invoke(main, ["--verbose", "--help"])
        assert result.exit_code == 0
