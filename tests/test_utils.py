"""Tests for aef_bng.utils - the settings and environment checks, not the Spark helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark.databricks.sql", reason="aef_bng.utils needs the spark extra")

from aef_bng.utils import (
    DatabricksSettings,
    NotInDatabricks,
    in_databricks,
    raise_error_if_not_in_databricks,
)


@pytest.mark.unit
class TestInDatabricks:
    """Runtime detection via DATABRICKS_RUNTIME_VERSION."""

    def test_true_on_a_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        assert in_databricks()
        raise_error_if_not_in_databricks()  # must not raise

    def test_false_locally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        assert not in_databricks()
        with pytest.raises(NotInDatabricks):
            raise_error_if_not_in_databricks()


@pytest.mark.unit
class TestDatabricksSettings:
    """Credentials from the environment (the .env file is ignored here)."""

    def test_reads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_pat = "not-a-real-pat"
        monkeypatch.setenv("DATABRICKS_HOST", "https://adb-1.azuredatabricks.net/")
        monkeypatch.setenv("DATABRICKS_TOKEN", fake_pat)
        monkeypatch.setenv("DATABRICKS_CLUSTER_ID", "0519-091045-abc123")
        settings = DatabricksSettings(_env_file=None)  # type: ignore[call-arg]

        assert settings.databricks_token == fake_pat
        assert settings.databricks_cluster_id == "0519-091045-abc123"
        assert settings.clean_host == "https://adb-1.azuredatabricks.net"

    @pytest.mark.parametrize(
        "host",
        [
            "adb-1.azuredatabricks.net",
            "https://adb-1.azuredatabricks.net",
            "adb-1.azuredatabricks.net/",
        ],
    )
    def test_host_with_scheme(self, monkeypatch: pytest.MonkeyPatch, host: str) -> None:
        monkeypatch.setenv("DATABRICKS_HOST", host)
        settings = DatabricksSettings(_env_file=None)  # type: ignore[call-arg]
        assert settings.host_with_scheme == "https://adb-1.azuredatabricks.net"

    def test_defaults_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CLUSTER_ID"):
            monkeypatch.delenv(var, raising=False)
        settings = DatabricksSettings(_env_file=None)  # type: ignore[call-arg]
        assert (settings.databricks_host, settings.databricks_token) == ("", "")
