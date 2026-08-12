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


@pytest.mark.unit
class TestBoundaryOptions:
    """Tests for the boundary-filter CLI options."""

    def test_spark_run_help_lists_boundary_options(self) -> None:
        """spark-run --help documents the boundary options."""
        result = CliRunner().invoke(main, ["spark-run", "--help"])
        assert result.exit_code == 0
        assert "--boundary-path" in result.output
        assert "--boundary-query" in result.output
        assert "--boundary-buffer-m" in result.output

    def test_empty_boundary_strings_disable_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty-string boundary params (DAB defaults) map to None in the config."""
        captured = {}
        monkeypatch.setattr(
            "aef_bng.spark.process_with_spark", lambda config: captured.update(config=config)
        )
        result = CliRunner().invoke(
            main,
            [
                "spark-run",
                "--bounds",
                "0,0,10000,10000",
                "--years",
                "2024",
                "--table-name",
                "cat.schema.table",
                "--boundary-path",
                "",
                "--boundary-query",
                "",
                "--boundary-buffer-m",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["config"].boundary_path is None
        assert captured["config"].boundary_query is None

    def test_boundary_options_reach_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-empty boundary options are passed through to the config."""
        captured = {}
        monkeypatch.setattr(
            "aef_bng.spark.process_with_spark", lambda config: captured.update(config=config)
        )
        result = CliRunner().invoke(
            main,
            [
                "spark-run",
                "--bounds",
                "0,0,10000,10000",
                "--years",
                "2024",
                "--table-name",
                "cat.schema.table",
                "--boundary-path",
                "/vol/bfe.geojson",
                "--boundary-query",
                "CTRY25NM in ['Wales']",
                "--boundary-buffer-m",
                "250",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["config"].boundary_path == "/vol/bfe.geojson"
        assert captured["config"].boundary_query == "CTRY25NM in ['Wales']"
        assert captured["config"].boundary_buffer_m == 250.0
